import ee
from utils import sync_fc_to_gee, check_task_status, make_asset_public, ee_initialize
from ..constants import LULC_V4


def get_farm_boundaries(asset_id):
    print("Inside get_farm_boundaries", asset_id)
    FARM_CLASSES = [8, 9, 10, 11, 13]
    SCRUB_CLASSES = [12, 7]
    FOREST_CLASS = [6]
    PLANTATION = [13]

    lulc = ee.Image(LULC_V4).select("predicted_label").rename("lulc").toInt()
    boundaries = ee.FeatureCollection(asset_id).filter(
        ee.Filter.neq("class", "plantation")
    )

    def get_feature_area(feature):
        area_in_sqm = feature.area()
        # area_in_hec = ee.Number(area_in_sqm).divide(ee.Number(10000))
        return feature.set("area_in_sqm", area_in_sqm)

    def get_distribution(feature):
        feature = ee.Feature(feature)
        feature = get_feature_area(feature)

        hist = lulc.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=feature.geometry(),
            scale=10,
        )

        # 2) Pull {class_str: count} from the 'lulc' band
        counts = ee.Dictionary(hist.get("lulc", ee.Dictionary({})))

        # 3) Total pixels inside this boundary
        total = ee.Number(
            ee.List(counts.values()).iterate(
                lambda v, acc: ee.Number(acc).add(ee.Number(v)), 0
            )
        )

        # 4) Sum helper for a list of numeric class IDs (convert each to string key)
        def sum_classes(class_list):
            class_list = ee.List(class_list)

            def _acc(c, acc):
                key = ee.Number(c).format()  # convert 8 -> "8"
                val = ee.Number(counts.get(key, 0))  # counts have string keys
                return ee.Number(acc).add(val)

            return ee.Number(class_list.iterate(_acc, 0))

        # 5) Compute farm % = classes 8,9,10,11 / total
        farm_count = sum_classes(FARM_CLASSES)
        plantation_count = sum_classes(PLANTATION)

        farm_pct = ee.Algorithms.If(
            ee.Number(total).gt(0), ee.Number(farm_count).divide(total).multiply(100), 0
        )
        plantation_pct = ee.Algorithms.If(
            ee.Number(total).gt(0),
            ee.Number(plantation_count).divide(total).multiply(100),
            0,
        )

        return feature.set("farm_pct", farm_pct)

    def simplify_boundary(feature):
        buf = 20
        tol = 5
        smooth = feature.geometry().buffer(buf).buffer(-buf + 1)
        simple = smooth.simplify(tol)
        return feature.setGeometry(simple)

    boundaries = boundaries.map(get_distribution)
    boundaries = boundaries.filter(ee.Filter.gt("farm_pct", 50))
    boundaries = boundaries.map(simplify_boundary)

    boundaries = boundaries.map(
        lambda f: f.set("geom_type", f.geometry().type())
    ).filter(ee.Filter.eq("geom_type", "Polygon"))

    boundaries = boundaries.filter(ee.Filter.gt("area_in_sqm", 50))

    description = asset_id.split("/")[-1]
    task_id = sync_fc_to_gee(
        boundaries, f"{description}_farm_pct", f"{asset_id}_farm_pct"
    )
    check_task_status([task_id])
    make_asset_public(f"{asset_id}_farm_pct")
