import ee
import ee
from pathlib import Path
import pandas as pd


from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
import time
from utils import ee_initialize


def store_emb_and_generate_LULC(AEZ_no):
    """
    Function to mask clouds based on the QA60 band of Sentinel SR data.
    param {ee.Image} image Input Sentinel SR image
    return {ee.Image} Cloudmasked Sentinel-2 image
    """

    def maskS2cloud(image):
        qa = image.select("QA60")
        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)

    """
    Function to clean builtup predictions using NDWI.
    """

    def ndwi_based_builtup_cleaning(
        roi_boundary, prediction_image, startDate, endDate, NDWI_threshold
    ):
        S2_ic = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .map(maskS2cloud)
        )

        if S2_ic.size().getInfo() != 0:
            S2_ic = S2_ic.map(
                lambda img: img.addBands(
                    img.normalizedDifference(["B3", "B8"]).rename("NDWI")
                )
            )
            NDWI_max_img = S2_ic.select("NDWI").max().clip(roi_boundary.geometry())

            corrected_water_img = prediction_image.select("predicted_label").where(
                prediction_image.select("predicted_label")
                .neq(0)
                .And(NDWI_max_img.gt(NDWI_threshold)),
                0,
            )
            return corrected_water_img
        else:
            print(
                "NDWI based builtup correction cannot be performed due to unavailability of Sentinel-2 data"
            )
            return prediction_image

    """
    Function to clean builtup predictions using NDVI.
    """

    def ndvi_based_builtup_cleaning(
        roi_boundary, prediction_image, startDate, endDate, NDVI_threshold
    ):
        S2_ic = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .map(maskS2cloud)
        )

        if S2_ic.size().getInfo() != 0:
            S2_ic = S2_ic.map(
                lambda img: img.addBands(
                    img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                )
            )
            NDVI_max_img = S2_ic.select("NDVI").max().clip(roi_boundary.geometry())

            corrected_builtup_img = prediction_image.select("predicted_label").where(
                prediction_image.select("predicted_label")
                .neq(0)
                .And(NDVI_max_img.gt(NDVI_threshold)),
                0,
            )
            return corrected_builtup_img
        else:
            print(
                "NDVI based builtup correction cannot be performed due to unavailability of Sentinel-2 data"
            )
            return prediction_image

    def get_builtup_prediction(roi_boundary, startDate, endDate):
        DW_ic = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .select("built", "label")
        )

        builtup_img = DW_ic.select("label").mode().rename("predicted_label")
        builtup_img = builtup_img.where(builtup_img.neq(6), 0)
        builtup_img = builtup_img.where(builtup_img.eq(6), 1)

        combined_builtup_img = builtup_img.clip(roi_boundary.geometry())

        ndwi_corrected_builtup_img = ndwi_based_builtup_cleaning(
            roi_boundary, combined_builtup_img, startDate, endDate, 0.25
        )
        ndvi_corrected_builtup_img = ndvi_based_builtup_cleaning(
            roi_boundary, ndwi_corrected_builtup_img, startDate, endDate, 0.5
        )

        return ndvi_corrected_builtup_img

    """
    Function to get the first date of the month of input start date and the last date of this month.
    It is used to advance the time range by 1 month in future code.
    """

    def get_start_and_end_of_month(input_date):
        year = input_date.get("year")
        month = input_date.get("month")

        start_of_month = ee.Date.fromYMD(year, month, 1)
        end_of_month = start_of_month.advance(1, "month").advance(-1, "day")

        return start_of_month, end_of_month

    """
    Function to get water body predictions in kharif using Sentinel-1 SAR data.
    """

    def get_kharif_bodies(roi_boundary, start_date, end_date):
        SAR_ic = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi_boundary)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        )

        kharif_month1_ic = SAR_ic.filterDate(start_date, end_date)
        kharif_month2_ic = SAR_ic.filterDate(
            start_date.advance(1, "month"), end_date.advance(1, "month")
        )
        kharif_month3_ic = SAR_ic.filterDate(
            start_date.advance(2, "month"), end_date.advance(2, "month")
        )
        kharif_month4_ic = SAR_ic.filterDate(
            start_date.advance(3, "month"), end_date.advance(3, "month")
        )

        ###
        ## Compute water mask
        ###
        kharif_month1_waterImg = (
            kharif_month1_ic.map(
                lambda img: img.addBands(img.select("VV").lt(-16).rename("Water"))
            )
            .select("Water")
            .mode()
        )
        kharif_month2_waterImg = (
            kharif_month2_ic.map(
                lambda img: img.addBands(img.select("VV").lt(-16).rename("Water"))
            )
            .select("Water")
            .mode()
        )
        kharif_month3_waterImg = (
            kharif_month3_ic.map(
                lambda img: img.addBands(img.select("VV").lt(-16).rename("Water"))
            )
            .select("Water")
            .mode()
        )
        kharif_month4_waterImg = (
            kharif_month4_ic.map(
                lambda img: img.addBands(img.select("VV").lt(-16).rename("Water"))
            )
            .select("Water")
            .mode()
        )

        kharif_ic = (
            ee.ImageCollection(kharif_month1_waterImg)
            .merge(kharif_month2_waterImg)
            .merge(kharif_month3_waterImg)
            .merge(kharif_month4_waterImg)
        )
        kharif_water_sum = kharif_ic.reduce(ee.Reducer.sum())
        kharif_water_mask = (
            kharif_water_sum.clip(roi_boundary.geometry()).gte(3).rename("Water")
        )

        return kharif_water_mask

    """
    Function to get water body predictions in Rabi using Dynamic World.
    """

    def get_rabi_bodies(roi_boundary, start_date, end_date):
        DW_ic = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(roi_boundary)
            .select(["label", "water"])
        )

        rabi_month1_ic = DW_ic.filterDate(
            start_date.advance(4, "month"), end_date.advance(4, "month")
        )
        rabi_month2_ic = DW_ic.filterDate(
            start_date.advance(5, "month"), end_date.advance(5, "month")
        )
        rabi_month3_ic = DW_ic.filterDate(
            start_date.advance(6, "month"), end_date.advance(6, "month")
        )
        rabi_month4_ic = DW_ic.filterDate(
            start_date.advance(7, "month"), end_date.advance(7, "month")
        )

        rabi_month1_img = (
            ee.Image(
                ee.Algorithms.If(
                    rabi_month1_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    rabi_month1_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        rabi_month2_img = (
            ee.Image(
                ee.Algorithms.If(
                    rabi_month2_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    rabi_month2_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        rabi_month3_img = (
            ee.Image(
                ee.Algorithms.If(
                    rabi_month3_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    rabi_month3_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        rabi_month4_img = (
            ee.Image(
                ee.Algorithms.If(
                    rabi_month4_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    rabi_month4_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        rabi_ic = (
            ee.ImageCollection(rabi_month1_img)
            .merge(rabi_month2_img)
            .merge(rabi_month3_img)
            .merge(rabi_month4_img)
        )
        rabi_water_mask = rabi_ic.reduce(ee.Reducer.sum()).gte(2).rename("Water")

        return rabi_water_mask

    """
    Function to get water body predictions in Zaid using Dynamic World.
    """

    def get_zaid_bodies(roi_boundary, start_date, end_date):
        DW_ic = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(roi_boundary)
            .select(["label", "water"])
        )

        zaid_month1_ic = DW_ic.filterDate(
            start_date.advance(8, "month"), end_date.advance(8, "month")
        )
        zaid_month2_ic = DW_ic.filterDate(
            start_date.advance(9, "month"), end_date.advance(9, "month")
        )
        zaid_month3_ic = DW_ic.filterDate(
            start_date.advance(10, "month"), end_date.advance(10, "month")
        )
        zaid_month4_ic = DW_ic.filterDate(
            start_date.advance(11, "month"), end_date.advance(11, "month")
        )

        zaid_month1_img = (
            ee.Image(
                ee.Algorithms.If(
                    zaid_month1_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    zaid_month1_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        zaid_month2_img = (
            ee.Image(
                ee.Algorithms.If(
                    zaid_month2_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    zaid_month2_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        zaid_month3_img = (
            ee.Image(
                ee.Algorithms.If(
                    zaid_month3_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    zaid_month3_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        zaid_month4_img = (
            ee.Image(
                ee.Algorithms.If(
                    zaid_month4_ic.size().eq(0),
                    ee.Image.constant(0).rename("label"),
                    zaid_month4_ic.select("label").mode().add(1),
                )
            )
            .clip(roi_boundary.geometry())
            .select("label")
            .eq(1)
        )

        zaid_ic = (
            ee.ImageCollection(zaid_month1_img)
            .merge(zaid_month2_img)
            .merge(zaid_month3_img)
            .merge(zaid_month4_img)
        )
        zaid_water_mask = zaid_ic.reduce(ee.Reducer.sum()).gte(2).rename("Water")

        return zaid_water_mask

    """
    Function to clean water predictions using NDWI.
    """

    def ndwi_based_water_cleaning(
        roi_boundary, prediction_image, startDate, endDate, NDWI_threshold
    ):
        S2_ic = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .map(maskS2cloud)
            .select(["B3", "B8"])
        )

        if S2_ic.size().getInfo() != 0:
            S2_ic = S2_ic.map(
                lambda img: img.addBands(
                    img.normalizedDifference(["B3", "B8"]).rename("NDWI")
                )
            )
            NDWI_max_img = S2_ic.select("NDWI").max().clip(roi_boundary.geometry())

            corrected_water_img = prediction_image.select("predicted_label").where(
                prediction_image.select("predicted_label")
                .neq(0)
                .And(NDWI_max_img.lt(NDWI_threshold)),
                0,
            )
            return corrected_water_img
        else:
            print(
                "NDWI based water correction cannot be performed due to unavailability of Sentinel-2 data"
            )
            return prediction_image

    """
    Main function to perform water classification
    """

    def get_water_prediction(roi_boundary, startDate, endDate):
        start_date, end_date = get_start_and_end_of_month(ee.Date(startDate))

        kharif_water_img = get_kharif_bodies(roi_boundary, start_date, end_date)
        rabi_water_img = get_rabi_bodies(roi_boundary, start_date, end_date)
        zaid_water_img = get_zaid_bodies(roi_boundary, start_date, end_date)

        kharif_water = kharif_water_img.select("Water").rename("predicted_label")
        rabi_water = rabi_water_img.select("Water").rename("predicted_label")
        zaid_water = zaid_water_img.select("Water").rename("predicted_label")
        combined_water_img = (
            kharif_water.where(kharif_water, 2)
            .where(rabi_water, 3)
            .where(zaid_water, 4)
        )

        # Clean the water predictions based on confidence and NDWI
        ndwi_corrected_img = ndwi_based_water_cleaning(
            roi_boundary, combined_water_img, startDate, endDate, 0.15
        )

        return ndwi_corrected_img

    def get_barrenland_prediction(roi_boundary, startDate, endDate):
        DW_ic = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .select("bare", "label")
        )

        bare_img = DW_ic.select("label").mode().rename("predicted_label")
        bare_img = bare_img.where(bare_img.neq(7), 0)

        bare_img = bare_img.clip(roi_boundary.geometry())

        return bare_img

    def fill_empty_bands(image):
        band_names = image.bandNames()
        zero_img = image.select(0).multiply(0).rename("constant").toDouble()
        zero_img_masked = zero_img.updateMask(zero_img)
        image = ee.Algorithms.If(
            ee.List(band_names).contains(ee.String("VV")),
            image,
            ee.Image(image).addBands(zero_img_masked.select("constant").rename("VV")),
        )
        image = ee.Algorithms.If(
            ee.List(band_names).contains(ee.String("VH")),
            image,
            ee.Image(image).addBands(zero_img_masked.select("constant").rename("VH")),
        )
        return image

    def Get_S1_ImageCollections(inputStartDate, inputEndDate, roi_boundary):
        S1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filterDate(inputStartDate, inputEndDate)
            .filterBounds(roi_boundary)
        )

        S1_processed = S1.map(fill_empty_bands)
        return S1_processed

    def GetVV_VH_image_datewise(S1_ic):
        def get_VV_VH_datewise(date):
            zero_img = S1_ic.first().select("VV", "VH").multiply(0)
            zero_img_masked = zero_img.updateMask(zero_img)

            subset_ic = S1_ic.select(["VV", "VH"]).filterDate(
                ee.Date(date), ee.Date(date).advance(16, "day")
            )
            image = ee.Algorithms.If(
                ee.Number(subset_ic.size()).gt(0),
                subset_ic.mean().set("system:time_start", ee.Date(date).millis()),
                zero_img.set("system:time_start", ee.Date(date).millis()),
            )

            return image

        return get_VV_VH_datewise

    def Get_S1_16Day_VV_VH_TimeSeries(inputStartDate, inputEndDate, S1_ic):
        startDate = datetime.strptime(inputStartDate, "%Y-%m-%d")
        endDate = datetime.strptime(inputEndDate, "%Y-%m-%d")

        date_list = pd.date_range(start=startDate, end=endDate, freq="16D").tolist()
        date_list = ee.List(
            [datetime.strftime(curr_date, "%Y-%m-%d") for curr_date in date_list]
        )

        S1_TS = ee.ImageCollection.fromImages(
            date_list.map(GetVV_VH_image_datewise(S1_ic))
        )
        return S1_TS

    def add_sarImg_timestamp(image):
        timeImage = image.metadata("system:time_start").rename("timestamp")
        timeImageMasked = timeImage.updateMask(image.mask().select(0))
        return image.addBands(timeImageMasked)

    def performInterpolation_sarTS(image):
        image = ee.Image(image)
        beforeImages = ee.List(image.get("before"))
        beforeMosaic = ee.ImageCollection.fromImages(beforeImages).mosaic()
        afterImages = ee.List(image.get("after"))
        afterMosaic = ee.ImageCollection.fromImages(afterImages).mosaic()

        # Interpolation formula
        # y = y1 + (y2-y1)*((t – t1) / (t2 – t1))
        # y = interpolated image
        # y1 = before image
        # y2 = after image
        # t = interpolation timestamp
        # t1 = before image timestamp
        # t2 = after image timestamp

        t1 = beforeMosaic.select("timestamp").rename("t1")
        t2 = afterMosaic.select("timestamp").rename("t2")
        t = image.metadata("system:time_start").rename("t")
        timeImage = ee.Image.cat([t1, t2, t])
        timeRatio = timeImage.expression(
            "(t - t1) / (t2 - t1)",
            {
                "t": timeImage.select("t"),
                "t1": timeImage.select("t1"),
                "t2": timeImage.select("t2"),
            },
        )

        interpolated = beforeMosaic.add(
            (afterMosaic.subtract(beforeMosaic).multiply(timeRatio))
        )
        result = image.unmask(interpolated)

        # Saketh
        # For data points on either end of time-series
        # Before or After mosaics may still have gaps (owing to few/no images in the window)
        # Simply fill with after mosaic (for first few data points) and before mosaic (for last few datapoints)
        fill_value = ee.ImageCollection([beforeMosaic, afterMosaic]).mosaic()
        result = result.unmask(fill_value)

        return result.copyProperties(image, ["system:time_start"])

    def interpolate_sar_timeseries(S1_TS):
        filtered = S1_TS.map(add_sarImg_timestamp)

        # Time window in which we are willing to look forward and backward for unmasked pixel in time series
        timeWindow = 120

        # Define a maxDifference filter to find all images within the specified days. Convert days to milliseconds.
        millis = ee.Number(timeWindow).multiply(1000 * 60 * 60 * 24)
        # Filter says that pick only those timestamps which lie between the 2 timestamps not more than millis difference apart
        maxDiffFilter = ee.Filter.maxDifference(
            difference=millis,
            leftField="system:time_start",
            rightField="system:time_start",
        )

        # Filter to find all images after a given image. Compare the image's timstamp against other images.
        # Images ahead of target image should have higher timestamp.
        lessEqFilter = ee.Filter.lessThanOrEquals(
            leftField="system:time_start", rightField="system:time_start"
        )

        # Similarly define this filter to find all images before a given image
        greaterEqFilter = ee.Filter.greaterThanOrEquals(
            leftField="system:time_start", rightField="system:time_start"
        )

        # Apply first join to find all images that are after the target image but within the timeWindow
        filter1 = ee.Filter.And(maxDiffFilter, lessEqFilter)
        join1 = ee.Join.saveAll(
            matchesKey="after", ordering="system:time_start", ascending=False
        )
        join1Result = join1.apply(
            primary=filtered, secondary=filtered, condition=filter1
        )

        # Apply first join to find all images that are after the target image but within the timeWindow
        filter2 = ee.Filter.And(maxDiffFilter, greaterEqFilter)
        join2 = ee.Join.saveAll(
            matchesKey="before", ordering="system:time_start", ascending=True
        )
        join2Result = join2.apply(
            primary=join1Result, secondary=join1Result, condition=filter2
        )

        interpolated_S1_TS = ee.ImageCollection(
            join2Result.map(performInterpolation_sarTS)
        )

        return interpolated_S1_TS

    def get_trained_model(training_data_assetpath):
        training_data = ee.FeatureCollection(training_data_assetpath)

        training_band_names = [
            "0_VV",
            "1_VV",
            "2_VV",
            "3_VV",
            "4_VV",
            "5_VV",
            "6_VV",
            "7_VV",
            "8_VV",
            "9_VV",
            "10_VV",
            "11_VV",
            "12_VV",
            "13_VV",
            "14_VV",
            "15_VV",
            "16_VV",
            "17_VV",
            "18_VV",
            "19_VV",
            "20_VV",
            "21_VV",
            "22_VV",
            "0_VH",
            "1_VH",
            "2_VH",
            "3_VH",
            "4_VH",
            "5_VH",
            "6_VH",
            "7_VH",
            "8_VH",
            "9_VH",
            "10_VH",
            "11_VH",
            "12_VH",
            "13_VH",
            "14_VH",
            "15_VH",
            "16_VH",
            "17_VH",
            "18_VH",
            "19_VH",
            "20_VH",
            "21_VH",
            "22_VH",
        ]

        trained_model = (
            ee.Classifier.smileRandomForest(numberOfTrees=100, seed=42)
            .setOutputMode("MULTIPROBABILITY")
            .train(
                features=training_data,
                classProperty="class",
                inputProperties=training_band_names,
            )
        )

        return trained_model

    def Get_slope(roi_boundary):
        dem = ee.Image("CGIAR/SRTM90_V4")
        slope = ee.Terrain.slope(dem)
        slope_image = slope.clip(roi_boundary.geometry())
        return slope_image

    """
    Function to clean cropland predictions using NDVI.
    """

    def ndvi_based_cropland_cleaning(
        roi_boundary, prediction_image, startDate, endDate, NDVI_threshold
    ):
        S2_ic = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .map(maskS2cloud)
            .select(["B4", "B8"])
        )

        if S2_ic.size().getInfo():
            S2_ic = S2_ic.map(
                lambda img: img.addBands(
                    img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                )
            )
            NDVI_max_img = S2_ic.select("NDVI").max().clip(roi_boundary.geometry())

            # Get barrenlands out as label 7
            corrected_cropland_img = prediction_image.select("predicted_label").where(
                (prediction_image.select("predicted_label").eq(5)).And(
                    NDVI_max_img.lt(NDVI_threshold)
                ),
                7,
            )

            return corrected_cropland_img
        else:
            print(
                "NDVI based cropland correction cannot be performed due to unavailability of Sentinel-2 data"
            )
            return prediction_image

    """
    Function to clean forest/tree predictions using NDVI.
    """

    def ndvi_based_forest_cleaning(
        roi_boundary, prediction_image, startDate, endDate, NDVI_threshold
    ):
        S2_ic = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .map(maskS2cloud)
            .select(["B4", "B8"])
        )

        if S2_ic.size().getInfo():
            S2_ic = S2_ic.map(
                lambda img: img.addBands(
                    img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                )
            )
            NDVI_max_img = S2_ic.select("NDVI").max().clip(roi_boundary.geometry())

            # Get barrenlands out as label 7
            corrected_forest_img = prediction_image.select("predicted_label").where(
                (prediction_image.select("predicted_label").eq(6)).And(
                    NDVI_max_img.lt(NDVI_threshold)
                ),
                7,
            )

            return corrected_forest_img
        else:
            print(
                "NDVI based forest correction cannot be performed due to unavailability of Sentinel-2 data"
            )
            return prediction_image

    def get_cropland_prediction(startDate, endDate, roi_boundary):
        training_data_assetpath = "projects/ee-indiasat/assets/Rasterized_Groundtruth/L2_TrainingData_SAR_TimeSeries_1Year"
        trained_model = get_trained_model(training_data_assetpath)

        S1_ic = Get_S1_ImageCollections(startDate, endDate, roi_boundary)
        S1_TS = Get_S1_16Day_VV_VH_TimeSeries(startDate, endDate, S1_ic)
        interpolated_S1_TS = interpolate_sar_timeseries(S1_TS)
        S1_TS_img = interpolated_S1_TS.toBands()
        S1_VV_TS_img = S1_TS_img.select([".*_VV"])
        S1_VH_TS_img = S1_TS_img.select([".*_VH"])

        training_band_names = [
            "0_VV",
            "1_VV",
            "2_VV",
            "3_VV",
            "4_VV",
            "5_VV",
            "6_VV",
            "7_VV",
            "8_VV",
            "9_VV",
            "10_VV",
            "11_VV",
            "12_VV",
            "13_VV",
            "14_VV",
            "15_VV",
            "16_VV",
            "17_VV",
            "18_VV",
            "19_VV",
            "20_VV",
            "21_VV",
            "22_VV",
            "0_VH",
            "1_VH",
            "2_VH",
            "3_VH",
            "4_VH",
            "5_VH",
            "6_VH",
            "7_VH",
            "8_VH",
            "9_VH",
            "10_VH",
            "11_VH",
            "12_VH",
            "13_VH",
            "14_VH",
            "15_VH",
            "16_VH",
            "17_VH",
            "18_VH",
            "19_VH",
            "20_VH",
            "21_VH",
            "22_VH",
        ]

        training_img = (
            S1_VV_TS_img.addBands(S1_VH_TS_img)
            .select(training_band_names)
            .clip(roi_boundary.geometry())
        )
        classified_image = training_img.classify(trained_model)

        roi_label_image = (
            classified_image.select(["classification"])
            .arrayArgmax()
            .arrayFlatten([["predicted_label"]])
        )
        roi_label_image = roi_label_image.add(5).toInt8()

        slope_img = Get_slope(roi_boundary)
        combined_img = roi_label_image.addBands(slope_img)

        # check if the slope is >20 deg, re-classify the pixel from cropland to non-cropland
        final_classified_img = combined_img.select(["predicted_label"]).where(
            combined_img.select("predicted_label")
            .eq(5)
            .And(combined_img.select("slope").gte(30)),
            6,
        )

        cropland_corrected_img = ndvi_based_cropland_cleaning(
            roi_boundary, final_classified_img, startDate, endDate, NDVI_threshold=0.15
        )
        forest_corrected_img = ndvi_based_forest_cleaning(
            roi_boundary, cropland_corrected_img, startDate, endDate, NDVI_threshold=0.3
        )

        return forest_corrected_img

    def dw_based_shrub_cleaning(
        roi_boundary, current_prediction_output, startDate, endDate
    ):
        DW_ic = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(roi_boundary)
            .filterDate(startDate, endDate)
            .select("shrub_and_scrub", "label")
        )

        bare_img = (
            DW_ic.select("label")
            .mode()
            .rename("predicted_label")
            .clip(roi_boundary.geometry())
        )
        corrected_output = current_prediction_output.where(
            (current_prediction_output.select("predicted_label").eq(8)).And(
                bare_img.select("predicted_label").eq(5)
            ),
            12,
        )

        return corrected_output

    chastainBandNames = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]

    # Regression model parameters from Table-4. MSI TOA reflectance as a function of OLI TOA reflectance.
    msiOLISlopes = [1.0946, 1.0043, 1.0524, 0.8954, 1.0049, 1.0002]
    msiOLIIntercepts = [-0.0107, 0.0026, -0.0015, 0.0033, 0.0065, 0.0046]

    # Regression model parameters from Table-5. MSI TOA reflectance as a function of ETM+ TOA reflectance.
    msiETMSlopes = [1.10601, 0.99091, 1.05681, 1.0045, 1.03611, 1.04011]
    msiETMIntercepts = [-0.0139, 0.00411, -0.0024, -0.0076, 0.00411, 0.00861]

    # Regression model parameters from Table-6. OLI TOA reflectance as a function of ETM+ TOA reflectance.
    oliETMSlopes = [1.03501, 1.00921, 1.01991, 1.14061, 1.04351, 1.05271]
    oliETMIntercepts = [-0.0055, -0.0008, -0.0021, -0.0163, -0.0045, 0.00261]

    # Construct dictionary to handle all pairwise combos
    chastainCoeffDict = {
        "MSI_OLI": [
            msiOLISlopes,
            msiOLIIntercepts,
            1,
        ],  # check what the third item corresponds to
        "MSI_ETM": [msiETMSlopes, msiETMIntercepts, 1],
        "OLI_ETM": [oliETMSlopes, oliETMIntercepts, 1],
        "OLI_MSI": [msiOLISlopes, msiOLIIntercepts, 0],
        "ETM_MSI": [msiETMSlopes, msiETMIntercepts, 0],
        "ETM_OLI": [oliETMSlopes, oliETMIntercepts, 0],
    }

    """
    Function to mask cloudy pixels in Landsat-7
    """

    def maskL7cloud(image):
        qa = image.select("QA_PIXEL")
        mask = qa.bitwiseAnd(1 << 4).eq(0)
        return (
            image.updateMask(mask)
            .select(["B1", "B2", "B3", "B4", "B5", "B7"])
            .rename("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2")
        )

    """
    Function to mask cloudy pixels in Landsat-8
    """

    def maskL8cloud(image):
        qa = image.select("QA_PIXEL")
        mask = qa.bitwiseAnd(1 << 4).eq(0)
        return (
            image.updateMask(mask)
            .select(["B2", "B3", "B4", "B5", "B6", "B7"])
            .rename("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2")
        )

    """
    Function to mask clouds using the quality band of Sentinel-2 TOA
    """

    def maskS2cloudTOA(image):
        qa = image.select("QA60")
        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return (
            image.updateMask(mask)
            .select(["B2", "B3", "B4", "B8", "B11", "B12"])
            .rename(["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"])
        )

    """
    Get Landsat and Sentinel image collections
    """

    def Get_L7_L8_S2_ImageCollections(inputStartDate, inputEndDate, roi_boundary):
        # ------ Landsat 7 TOA
        L7 = (
            ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA")
            .filterDate(inputStartDate, inputEndDate)
            .filterBounds(roi_boundary)
            .map(maskL7cloud)
        )
        # print('\n Original Landsat 7 TOA dataset: \n',L7.limit(1).getInfo())
        # print('Number of images in Landsat 7 TOA dataset: \t',L7.size().getInfo())

        # ------ Landsat 8 TOA
        L8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
            .filterDate(inputStartDate, inputEndDate)
            .filterBounds(roi_boundary)
            .map(maskL8cloud)
        )
        # print('\n Original Landsat 8 TOA dataset: \n', L8.limit(1).getInfo())
        # print('Number of images in Landsat 8 TOA dataset: \t',L8.size().getInfo())

        # ------ Sentinel-2 TOA
        S2 = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterDate(inputStartDate, inputEndDate)
            .filterBounds(roi_boundary)
            .map(maskS2cloudTOA)
        )
        # print('\n Original Sentinel-2 TOA dataset: \n',S2.limit(1).getInfo())
        # print('Number of images in Sentinel 2 TOA dataset: \t',S2.size().getInfo())

        return L7, L8, S2

    """
    Function to apply model in one direction
    """

    def dir0Regression(img, slopes, intercepts):
        return img.select(chastainBandNames).multiply(slopes).add(intercepts)

    """
    Applying the model in the opposite direction
    """

    def dir1Regression(img, slopes, intercepts):
        return img.select(chastainBandNames).subtract(intercepts).divide(slopes)

    """
    Function to correct one sensor to another
    """

    def harmonizationChastain(img, fromSensor, toSensor):
        # Get the model for the given from and to sensor
        comboKey = fromSensor.upper() + "_" + toSensor.upper()
        coeffList = chastainCoeffDict[comboKey]
        slopes = coeffList[0]
        intercepts = coeffList[1]
        direction = ee.Number(coeffList[2])

        # Apply the model in the respective direction
        out = ee.Algorithms.If(
            direction.eq(0),
            dir0Regression(img, slopes, intercepts),
            dir1Regression(img, slopes, intercepts),
        )
        return (
            ee.Image(out).copyProperties(img).copyProperties(img, ["system:time_start"])
        )

    """
    Calibrate Landsat-8 (OLI) and Sentinel-2 (MSI) to Landsat-7 (ETM+)
    """

    def Harmonize_L7_L8_S2(L7, L8, S2):
        # harmonization
        harmonized_L8 = L8.map(lambda img: harmonizationChastain(img, "OLI", "ETM"))
        harmonized_S2 = S2.map(lambda img: harmonizationChastain(img, "MSI", "ETM"))

        # Merge harmonized landsat-8 and sentinel-2 to landsat-7 image collection
        harmonized_LandsatSentinel_ic = ee.ImageCollection(
            L7.merge(harmonized_L8).merge(harmonized_S2)
        )
        # print(harmonized_LandsatSentinel_ic.size().getInfo())
        return harmonized_LandsatSentinel_ic

    """
    Add NDVI band to harmonized image collection
    """

    def addNDVI(image):
        return image.addBands(
            image.normalizedDifference(["NIR", "RED"]).rename("NDVI")
        ).float()

    """
    Function definitions to get NDVI values at each 16-day composites
    """

    def Get_NDVI_image_datewise(harmonized_LS_ic, roi_boundary):
        def get_NDVI_datewise(date):
            empty_band_image = (
                ee.Image(0)
                .float()
                .rename(["NDVI"])
                .updateMask(ee.Image(0).clip(roi_boundary))
            )
            return (
                harmonized_LS_ic.select(["NDVI"])
                .filterDate(ee.Date(date), ee.Date(date).advance(16, "day"))
                .merge(empty_band_image)
                .median()
                .set("system:time_start", ee.Date(date).millis())
            )

        return get_NDVI_datewise

    def Get_LS_16Day_NDVI_TimeSeries(
        inputStartDate, inputEndDate, harmonized_LS_ic, roi_boundary
    ):
        startDate = datetime.strptime(inputStartDate, "%Y-%m-%d")
        endDate = datetime.strptime(inputEndDate, "%Y-%m-%d")

        date_list = pd.date_range(start=startDate, end=endDate, freq="16D").tolist()
        date_list = ee.List(
            [datetime.strftime(curr_date, "%Y-%m-%d") for curr_date in date_list]
        )

        LSC = ee.ImageCollection.fromImages(
            date_list.map(Get_NDVI_image_datewise(harmonized_LS_ic, roi_boundary))
        )

        return LSC

    """
    Pair available LSC and modis values for each time stamp.
    """

    def pairLSModis(lsRenameBands):
        def pair(feature):
            date = ee.Date(feature.get("system:time_start"))
            startDateT = date.advance(-8, "day")
            endDateT = date.advance(8, "day")

            # ------ MODIS VI ( We can add EVI to the band list later )
            modis = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(startDateT, endDateT)
                .select(["NDVI", "SummaryQA"])
                .filterBounds(roi_boundary)
                .median()
                .rename(["NDVI_modis", "SummaryQA_modis"])
            )

            return feature.rename(lsRenameBands).addBands(modis)

        return pair

    """
    Function to get Pearson Correlation Coffecient to perform GapFilling
    """

    def get_Pearson_Correlation_Coefficients(
        LSC_modis_paired_ic, roi_boundary, bandList
    ):
        corr = (
            LSC_modis_paired_ic.filterBounds(roi_boundary)
            .select(bandList)
            .toArray()
            .arrayReduce(
                reducer=ee.Reducer.pearsonsCorrelation(), axes=[0], fieldAxis=1
            )
            .arrayProject([1])
            .arrayFlatten([["c", "p"]])
        )
        return corr

    """Use print(...) to write to this console.
    Fill gaps in LSC timeseries using modis data
    """

    def gapfillLSM(LSC_modis_regression_model, LSC_bandName, modis_bandName):
        def peformGapfilling(image):
            offset = LSC_modis_regression_model.select("offset")
            scale = LSC_modis_regression_model.select("scale")
            nodata = -1

            lsc_image = image.select(LSC_bandName)
            modisfit = image.select(modis_bandName).multiply(scale).add(offset)

            mask = (
                lsc_image.mask()
            )  # update mask needs an input (no default input from the API document)
            gapfill = lsc_image.unmask(nodata)
            gapfill = gapfill.where(mask.Not(), modisfit)

            """
            in SummaryQA,
            0: Good data, use with confidence
            1: Marginal data, useful but look at detailed QA for more information
            2: Pixel covered with snow/ice
            3: Pixel is cloudy
            """
            qc_m = image.select("SummaryQA_modis").unmask(
                3
            )  # missing value is grouped as cloud
            w_m = modisfit.mask().rename("w_m").where(qc_m.eq(0), 0.8)  # default is 0.8
            w_m = w_m.where(qc_m.eq(1), 0.5)  # Marginal
            w_m = w_m.where(qc_m.gte(2), 0.2)  # snow/ice or cloudy

            # make sure these modis values are read where there is missing data from LandSat, Sentinel
            w_l = gapfill.mask()  # default is 1
            w_l = w_l.where(mask.Not(), w_m)

            return gapfill.addBands(w_l).rename(
                ["gapfilled_" + LSC_bandName, "SummaryQA"]
            )  # have NDVI from modis and a summary of clarity for each

        return peformGapfilling

    """
    Function to combine LSC with Modis data
    """

    def Combine_LS_Modis(LSC):
        lsRenameBands = (
            ee.Image(LSC.first())
            .bandNames()
            .map(lambda band: ee.String(band).cat("_lsc"))
        )
        LSC_modis_paired_ic = LSC.map(pairLSModis(lsRenameBands))

        # Output contains scale, offset i.e. two bands
        LSC_modis_regression_model_NDVI = LSC_modis_paired_ic.select(
            ["NDVI_modis", "NDVI_lsc"]
        ).reduce(ee.Reducer.linearFit())

        corr_NDVI = get_Pearson_Correlation_Coefficients(
            LSC_modis_paired_ic, roi_boundary, ["NDVI_modis", "NDVI_lsc"]
        )
        LSMC_NDVI = LSC_modis_paired_ic.map(
            gapfillLSM(LSC_modis_regression_model_NDVI, "NDVI_lsc", "NDVI_modis")
        )

        return LSMC_NDVI

    """
    Mask out low quality pixels
    """

    def mask_low_QA(lsmc_image):
        low_qa = lsmc_image.select("SummaryQA").neq(0.2)
        return lsmc_image.updateMask(low_qa).copyProperties(
            lsmc_image, ["system:time_start"]
        )

    """
    Add image timestamp to each image in time series
    """

    def add_timestamp(image):
        timeImage = image.metadata("system:time_start").rename("timestamp")
        timeImageMasked = timeImage.updateMask(image.mask().select(0))
        return image.addBands(timeImageMasked)

    """
    Perform linear interpolation on missing values
    """

    def performInterpolation(image):
        image = ee.Image(image)
        beforeImages = ee.List(image.get("before"))
        beforeMosaic = ee.ImageCollection.fromImages(beforeImages).mosaic()
        afterImages = ee.List(image.get("after"))
        afterMosaic = ee.ImageCollection.fromImages(afterImages).mosaic()

        # Interpolation formula
        # y = y1 + (y2-y1)*((t – t1) / (t2 – t1))
        # y = interpolated image
        # y1 = before image
        # y2 = after image
        # t = interpolation timestamp
        # t1 = before image timestamp
        # t2 = after image timestamp

        t1 = beforeMosaic.select("timestamp").rename("t1")
        t2 = afterMosaic.select("timestamp").rename("t2")
        t = image.metadata("system:time_start").rename("t")
        timeImage = ee.Image.cat([t1, t2, t])
        timeRatio = timeImage.expression(
            "(t - t1) / (t2 - t1)",
            {
                "t": timeImage.select("t"),
                "t1": timeImage.select("t1"),
                "t2": timeImage.select("t2"),
            },
        )

        interpolated = beforeMosaic.add(
            (afterMosaic.subtract(beforeMosaic).multiply(timeRatio))
        )
        result = image.unmask(interpolated)
        fill_value = ee.ImageCollection([beforeMosaic, afterMosaic]).mosaic()
        result = result.unmask(fill_value)

        return result.copyProperties(image, ["system:time_start"])

    def interpolate_timeseries(S1_TS):
        lsmc_masked = S1_TS.map(mask_low_QA)
        filtered = lsmc_masked.map(add_timestamp)

        # Time window in which we are willing to look forward and backward for unmasked pixel in time series
        timeWindow = 120

        # Define a maxDifference filter to find all images within the specified days. Convert days to milliseconds.
        millis = ee.Number(timeWindow).multiply(1000 * 60 * 60 * 24)
        # Filter says that pick only those timestamps which lie between the 2 timestamps not more than millis difference apart
        maxDiffFilter = ee.Filter.maxDifference(
            difference=millis,
            leftField="system:time_start",
            rightField="system:time_start",
        )

        # Filter to find all images after a given image. Compare the image's timstamp against other images.
        # Images ahead of target image should have higher timestamp.
        lessEqFilter = ee.Filter.lessThanOrEquals(
            leftField="system:time_start", rightField="system:time_start"
        )

        # Similarly define this filter to find all images before a given image
        greaterEqFilter = ee.Filter.greaterThanOrEquals(
            leftField="system:time_start", rightField="system:time_start"
        )

        # Apply first join to find all images that are after the target image but within the timeWindow
        filter1 = ee.Filter.And(maxDiffFilter, lessEqFilter)
        join1 = ee.Join.saveAll(
            matchesKey="after", ordering="system:time_start", ascending=False
        )
        join1Result = join1.apply(
            primary=filtered, secondary=filtered, condition=filter1
        )

        # Apply first join to find all images that are after the target image but within the timeWindow
        filter2 = ee.Filter.And(maxDiffFilter, greaterEqFilter)
        join2 = ee.Join.saveAll(
            matchesKey="before", ordering="system:time_start", ascending=True
        )
        join2Result = join2.apply(
            primary=join1Result, secondary=join1Result, condition=filter2
        )

        interpolated_S1_TS = ee.ImageCollection(join2Result.map(performInterpolation))

        return interpolated_S1_TS

    """
    Function Definition to get Padded NDVI LSMC timeseries image for a given ROI
    """

    def Get_Padded_NDVI_TS_Image(startDate, endDate, roi_boundary):
        L7, L8, S2 = Get_L7_L8_S2_ImageCollections(startDate, endDate, roi_boundary)

        harmonized_LS_ic = Harmonize_L7_L8_S2(L7, L8, S2)
        harmonized_LS_ic = harmonized_LS_ic.map(addNDVI)
        LSC = Get_LS_16Day_NDVI_TimeSeries(
            startDate, endDate, harmonized_LS_ic, roi_boundary
        )
        LSMC_NDVI = Combine_LS_Modis(LSC)
        Interpolated_LSMC_NDVI = interpolate_timeseries(LSMC_NDVI)
        final_LSMC_NDVI_TS = Interpolated_LSMC_NDVI.select(
            ["gapfilled_NDVI_lsc"]
        ).toBands()
        final_LSMC_NDVI_TS = final_LSMC_NDVI_TS.clip(roi_boundary)

        input_bands = final_LSMC_NDVI_TS.bandNames()
        return final_LSMC_NDVI_TS, input_bands

    """
    Function definition to compute euclidean distance to each cluster centroid
    features ---> clusters
    flattened ---> time series image clipped to target area
    input_bands ---> band names for time series image
    studyarea ---> geometry of region of interest
    """

    # Function to get distances as required from each pixel to each cluster centroid
    def Get_Euclidean_Distance(
        cluster_centroids, roi_timeseries_img, input_bands, roi_boundary
    ):

        def wrapper(curr_centroid):
            temp_img = ee.Image()
            curr_centroid = ee.Feature(curr_centroid).setGeometry(roi_boundary)
            temp_fc = ee.FeatureCollection([curr_centroid])
            class_img = (
                temp_fc.select(["class"])
                .reduceToImage(["class"], ee.Reducer.first())
                .rename(["class"])
            )

            def create_img(band_name):
                return (
                    temp_fc.select([band_name])
                    .reduceToImage([band_name], ee.Reducer.first())
                    .rename([band_name])
                )

            temp_img = input_bands.map(create_img)
            empty = ee.Image()
            temp_img = ee.Image(
                temp_img.iterate(lambda img, prev: ee.Image(prev).addBands(img), empty)
            )

            temp_img = temp_img.select(temp_img.bandNames().remove("constant"))
            distance = roi_timeseries_img.spectralDistance(temp_img, "sed")
            confidence = ee.Image(1.0).divide(distance).rename(["confidence"])
            distance = distance.addBands(confidence)
            return distance.addBands(class_img.rename(["class"])).set(
                "class", curr_centroid.get("class")
            )

        return cluster_centroids.map(wrapper)

    """
    Function definition to get final prediction image from distance images
    """

    def Get_final_prediction_image(distance_imgs_list):
        # Denominator is an image that is sum of all confidences to each cluster
        sum_of_distances = (
            ee.ImageCollection(distance_imgs_list).select(["confidence"]).sum().unmask()
        )
        distance_imgs_ic = ee.ImageCollection(distance_imgs_list).select(
            ["distance", "class"]
        )

        # array is an image where distance band is an array of distances to each cluster centroid and class band is an array of classes associated with each cluster
        array_img = ee.ImageCollection(distance_imgs_ic).toArray()

        axes = {"image": 0, "band": 1}
        sort = array_img.arraySlice(axes["band"], 0, 1)
        sorted = array_img.arraySort(sort)

        # take the first image only
        values = sorted.arraySlice(axes["image"], 0, 1)
        # convert back to an image
        min = values.arrayProject([axes["band"]]).arrayFlatten([["distance", "class"]])
        # Extract the hard classification
        predicted_class_img = min.select(1)
        predicted_class_img = predicted_class_img.rename(["predicted_label"])

        return predicted_class_img

    ## My Helper Functions
    def change_clusters(cluster_centroids):
        size = cluster_centroids.size().getInfo()
        features = []
        for i in range(size):
            features.append(
                ee.Feature(cluster_centroids.toList(size).get(i)).set("class", 13 + i)
            )
        return ee.FeatureCollection(features)

    def get_cropping_frequency(roi_boundary, startDate, endDate):
        cluster_centroids = ee.FeatureCollection(
            "projects/ee-indiasat/assets/L3_LULC_Clusters/Final_Level3_PanIndia_Clusters"
        )
        ignore_clusters = [12]  # remove invalid clusters
        cluster_centroids = cluster_centroids.filter(
            ee.Filter.Not(ee.Filter.inList("class", ignore_clusters))
        )

        final_LSMC_NDVI_TS, input_bands = Get_Padded_NDVI_TS_Image(
            startDate, endDate, roi_boundary
        )
        distance_imgs_list = Get_Euclidean_Distance(
            cluster_centroids, final_LSMC_NDVI_TS, input_bands, roi_boundary
        )
        final_classified_img = Get_final_prediction_image(distance_imgs_list)
        ### adding Cluster values after 12
        # cluster_centroids = change_clusters(cluster_centroids)
        distance_imgs_list = Get_Euclidean_Distance(
            cluster_centroids, final_LSMC_NDVI_TS, input_bands, roi_boundary
        )
        final_cluster_classified_img = Get_final_prediction_image(distance_imgs_list)
        final_cluster_classified_img = final_cluster_classified_img.rename(
            ["predicted_cluster"]
        )
        final_classified_img = final_classified_img.addBands(
            final_cluster_classified_img
        )
        return final_classified_img, final_LSMC_NDVI_TS

    def split_roi(fc):
        roi = ee.Feature(fc.first()).geometry().dissolve()

        # 3) Compute axis-aligned bounding box and its extents.
        bbox = roi.bounds(1)
        ring = ee.List(ee.List(bbox.coordinates()).get(0))
        xs = ring.map(lambda p: ee.Number(ee.List(p).get(0)))
        ys = ring.map(lambda p: ee.Number(ee.List(p).get(1)))
        minX = ee.Number(xs.reduce(ee.Reducer.min()))
        maxX = ee.Number(xs.reduce(ee.Reducer.max()))
        minY = ee.Number(ys.reduce(ee.Reducer.min()))
        maxY = ee.Number(ys.reduce(ee.Reducer.max()))

        width = maxX.subtract(minX)
        height = maxY.subtract(minY)
        splitVertically = width.gte(
            height
        )  # If wider than tall → vertical split; else horizontal.

        xMid = minX.add(maxX).divide(2)
        yMid = minY.add(maxY).divide(2)

        # 4) Build two half-rectangles that cover the bbox.
        leftRect = ee.Geometry.Rectangle(
            [minX, minY, xMid, maxY], proj=None, geodesic=False
        )
        rightRect = ee.Geometry.Rectangle(
            [xMid, minY, maxX, maxY], proj=None, geodesic=False
        )
        bottomRect = ee.Geometry.Rectangle(
            [minX, minY, maxX, yMid], proj=None, geodesic=False
        )
        topRect = ee.Geometry.Rectangle(
            [minX, yMid, maxX, maxY], proj=None, geodesic=False
        )

        # 5) Intersect ROI with the chosen halves to get two parts.
        part1 = ee.Geometry(
            ee.Algorithms.If(
                splitVertically,
                roi.intersection(leftRect, 1),
                roi.intersection(bottomRect, 1),
            )
        )
        part2 = ee.Geometry(
            ee.Algorithms.If(
                splitVertically,
                roi.intersection(rightRect, 1),
                roi.intersection(topRect, 1),
            )
        )

        # 6) Wrap as a FeatureCollection (two parts).
        parts_fc = ee.FeatureCollection(
            [ee.Feature(part1).set({"part": 1}), ee.Feature(part2).set({"part": 2})]
        )

        # (Optional) sanity check: areas (m^2)
        print(
            "Areas (m^2):",
            ee.Dictionary(
                {"total": roi.area(1), "part1": part1.area(1), "part2": part2.area(1)}
            ).getInfo(),
        )
        return parts_fc

    roi_boundary = ee.FeatureCollection("users/mtpictd/agro_eco_regions").filter(
        ee.Filter.eq("ae_regcode", AEZ_no)
    )
    filename_prefix = "AEZ_" + str(AEZ_no)

    mapping = {"farm": 1, "plantation": 2, "scrubland": 3, "rest": 0}

    def write_to_gee(fc, asset_name):

        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description="Working on " + asset_name.split("/")[-1],
            assetId=asset_name,
        )
        task.start()
        print(f"Started export to GEE asset: {asset_name}")
        while True:
            status = task.status()
            state = status.get("state", "UNKNOWN")
            print(f"Export status: {state}")
            if state in ["COMPLETED", "FAILED", "CANCELLED"]:
                break
            time.sleep(30)
        if state == "COMPLETED":
            print(f"Export to GEE asset {asset_name} completed successfully.")
        else:
            print(f"Export to GEE asset {asset_name} failed with status: {state}")

    def export_samples_in_chunks(image, samples, chunk_size, asset_prefix):
        total = samples.size().getInfo()
        assets = []
        for index, i in enumerate(range(0, total, chunk_size)):
            # Get a chunk of features
            chunk = ee.FeatureCollection(samples.toList(chunk_size, i))
            # Sample the image at these points
            chunk_samples = image.sampleRegions(
                collection=chunk, scale=10, geometries=True
            )
            # Export this chunk
            asset_id = f"{asset_prefix}_emb_{index}"
            write_to_gee(chunk_samples, asset_id)
            assets.append(asset_id)
        return assets

    def stratified_sample_by_tile_label(
        fc: ee.FeatureCollection,
        tile_prop: str = "tile_index",
        label_prop: str = "label",
        fraction: float = 0.2,
        seed: int = 1,
        min_per_group: int = 1,
        drop_random: bool = True,
    ) -> ee.FeatureCollection:
        """
        Stratified random sample (by fraction) from each (tile_prop, label_prop) stratum.

        Returns:
            ee.FeatureCollection with a per-stratum fraction (ceil) and >= min_per_group (bounded by stratum size).
        """
        # Clamp fraction
        fraction = max(0.0, min(1.0, float(fraction)))

        # All unique (tile, label) pairs
        groups = fc.distinct([tile_prop, label_prop]).select([tile_prop, label_prop])

        empty_fc = ee.FeatureCollection([])
        empty_list = ee.List([])

        def _iterate(group, acc_list):
            group = ee.Feature(group)
            acc_list = ee.List(acc_list)

            tile_val = group.get(tile_prop)
            label_val = group.get(label_prop)

            subset = fc.filter(ee.Filter.eq(tile_prop, tile_val)).filter(
                ee.Filter.eq(label_prop, label_val)
            )

            size = ee.Number(subset.size())
            n = size.multiply(fraction).ceil()  # exact-ish per-group fraction
            n = n.max(ee.Number(min_per_group)).min(size)  # enforce min, cap by size

            sampled = ee.FeatureCollection(
                ee.Algorithms.If(
                    size.gt(0),
                    subset.randomColumn("rand", seed).sort("rand").limit(n),
                    empty_fc,
                )
            )

            if drop_random:
                # Remove 'rand' safely without touching .first() on empty collections
                sampled = sampled.map(
                    lambda f: f.select(ee.List(f.propertyNames()).remove("rand"))
                )

            # Append sampled features (as a list) to the accumulator list
            return acc_list.cat(sampled.toList(sampled.size()))

        # Accumulate a List<Feature>, then wrap once as a FeatureCollection
        sampled_list = ee.List(groups.iterate(_iterate, empty_list))
        sampled_fc = ee.FeatureCollection(sampled_list)
        return sampled_fc

    emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    emb_2025 = (
        emb.filterDate("2024-01-01", "2025-01-01").filterBounds(roi_boundary).mosaic()
    )
    emb_2024 = (
        emb.filterDate("2023-01-01", "2024-01-01").filterBounds(roi_boundary).mosaic()
    )
    emb_2023 = (
        emb.filterDate("2022-01-01", "2023-01-01").filterBounds(roi_boundary).mosaic()
    )

    samples = ee.FeatureCollection(
        "projects/ee-amanverma/assets/lulc_v4/gee_samples_all"
    ).filter(ee.Filter.eq("aez_no", AEZ_no))
    samples = stratified_sample_by_tile_label(
        samples, fraction=0.33, seed=42, min_per_group=50
    )
    # import ipdb
    # ipdb.set_trace()
    sample_assets_2025 = export_samples_in_chunks(
        emb_2025,  # your image
        samples,  # your FeatureCollection
        chunk_size=20000,  # adjust as needed (try 1000-10000)
        asset_prefix=f"projects/ee-amanverma/assets/lulc_v4/AEZ_{AEZ_no}_samples_from_local_2025",
    )
    sample_assets_2024 = export_samples_in_chunks(
        emb_2024,  # your image
        samples,  # your FeatureCollection
        chunk_size=20000,  # adjust as needed (try 1000-10000)
        asset_prefix=f"projects/ee-amanverma/assets/lulc_v4/AEZ_{AEZ_no}_samples_from_local_2024",
    )
    sample_assets_2023 = export_samples_in_chunks(
        emb_2023,  # your image
        samples,  # your FeatureCollection
        chunk_size=20000,  # adjust as needed (try 1000-10000)
        asset_prefix=f"projects/ee-amanverma/assets/lulc_v4/AEZ_{AEZ_no}_samples_from_local_2023",
    )
    sample_assets = sample_assets_2025 + sample_assets_2024 + sample_assets_2023

    # Read each asset path as a FeatureCollection and merge them
    def get_samples(roi):
        all_samples = ee.FeatureCollection([])  # start with empty
        for asset_path in sample_assets:
            fc = ee.FeatureCollection(asset_path)
            all_samples = all_samples.merge(fc)
        all_samples = all_samples.filterBounds(roi)
        return all_samples

    def get_classifier(bandnames, roi):
        # import ipdb

        # ipdb.set_trace()
        classifier = ee.Classifier.smileRandomForest(numberOfTrees=100, seed=42).train(
            features=get_samples(roi).limit(100000),
            classProperty="label",
            inputProperties=bandnames,
        )
        return classifier

    def get_emb_date(date):
        date = datetime.strptime(date, "%Y-%m-%d")
        next_year = date.year + 1
        return f"{next_year}-01-01"

    def get_lulc(roi_boundary, parts=""):
        startDate = "2017-07-01"
        endDate = "2025-07-01"

        L1_asset_new = []
        final_output_filename_array_new = []
        final_output_assetid_array_new = []
        crop_freq_array = []

        scale = 10

        loopStart = startDate
        loopEnd = (datetime.strptime(endDate, "%Y-%m-%d")).strftime("%Y-%m-%d")

        while loopStart != loopEnd:
            currStartDate = datetime.strptime(loopStart, "%Y-%m-%d")
            currEndDate = currStartDate + relativedelta(years=1) - timedelta(days=1)
            loopStart = (currStartDate + relativedelta(years=1)).strftime("%Y-%m-%d")

            currStartDate = currStartDate.strftime("%Y-%m-%d")
            currEndDate = currEndDate.strftime("%Y-%m-%d")

            print(
                "\n EXECUTING LULC PREDICTION FOR ",
                currStartDate,
                " TO ",
                currEndDate,
                "\n",
            )

            curr_filename = filename_prefix + "_" + currStartDate + "_" + currEndDate

            if datetime.strptime(currStartDate, "%Y-%m-%d").year < 2017:
                print(
                    "To generate LULC output of year ",
                    datetime.strptime(currStartDate, "%Y-%m-%d").year,
                    " , go to cell-LULC execution for years before 2017",
                )
                continue

            # LULC prediction code
            bu_image = get_builtup_prediction(roi_boundary, currStartDate, currEndDate)
            water_image = get_water_prediction(roi_boundary, currStartDate, currEndDate)
            combined_water_builtup_img = bu_image.where(
                bu_image.select("predicted_label").eq(0), water_image
            )
            bare_image = get_barrenland_prediction(
                roi_boundary, currStartDate, currEndDate
            )
            combined_water_builtup_barren_img = combined_water_builtup_img.where(
                combined_water_builtup_img.select("predicted_label").eq(0), bare_image
            )

            cropping_frequency_img, time_series_data = get_cropping_frequency(
                roi_boundary, currStartDate, currEndDate
            )

            embStartDate = get_emb_date(currStartDate)
            embEndDate = get_emb_date(currEndDate)
            embeddings = (
                ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                .filterDate(embStartDate, embEndDate)
                .filterBounds(roi_boundary)
                .mosaic()
            )
            classifier = get_classifier(embeddings.bandNames(), roi_boundary)

            scrubland_farm_raster = embeddings.classify(classifier).rename(
                "predicted_label"
            )
            scrubland_farm_raster = (
                scrubland_farm_raster.where(
                    scrubland_farm_raster.eq(mapping["farm"]), 5
                )
                .where(scrubland_farm_raster.eq(mapping["scrubland"]), 12)
                .where(scrubland_farm_raster.eq(mapping["plantation"]), 13)
            )
            combined_img = combined_water_builtup_barren_img.where(
                combined_water_builtup_barren_img.select("predicted_label").eq(0),
                scrubland_farm_raster,
            )

            cropland_image = get_cropland_prediction(
                currStartDate, currEndDate, roi_boundary
            )
            tree_image = cropland_image.where(
                cropland_image.select("predicted_label").eq(5), 12
            )
            combined_img = combined_img.where(
                combined_img.select("predicted_label").eq(12), tree_image
            )
            final_lulc_img = combined_img.addBands(
                ee.Image.constant(-1).rename(["predicted_cluster"])
            ).where(
                combined_img.select("predicted_label").eq(5), cropping_frequency_img
            )

            final_output_filename = curr_filename + "_LULCmap_" + str(scale) + "m"
            final_output_assetid = (
                "projects/ee-amanverma/assets/lulc_v4/LULC_Version2_Outputs_NewHierarchy/"
                + final_output_filename
            )

            final_output_filename_array_new.append(final_output_filename)
            final_output_assetid_array_new.append(final_output_assetid)
            L1_asset_new.append(final_lulc_img)
            # displayMap(roi_boundary, final_lulc_img.select('predicted_label'))

            task = ee.batch.Export.image.toAsset(
                image=final_lulc_img,  # .select("predicted_label"),
                description="lulc_"
                + filename_prefix
                + "_v4"
                + "_"
                + currStartDate
                + "_"
                + currEndDate,
                assetId="projects/ee-amanverma/assets/lulc_v4/"
                + filename_prefix
                + "_v4_temporal_3years_"
                + parts
                + "_"
                + currStartDate
                + "_"
                + currEndDate,
                pyramidingPolicy={"predicted_label": "mode"},
                scale=10,
                maxPixels=1e13,
                crs="EPSG:4326",
            )
            task.start()
        return

    if AEZ_no in [2, 6]:
        parts_fc = split_roi(roi_boundary)
        part1 = parts_fc.filter(ee.Filter.eq("part", 1))
        part2 = parts_fc.filter(ee.Filter.eq("part", 2))
        get_lulc(part1, "_part1")
        get_lulc(part2, "_part2")
    else:
        get_lulc(roi_boundary, "")


if __name__ == "__main__":
    print("Inside store_embeddings_run_lulcv4 script")
    ee_initialize("aman")
    for AEZ_no in [3]:
        store_emb_and_generate_LULC(AEZ_no)
