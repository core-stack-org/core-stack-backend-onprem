from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status


from compute.compute_layers import scrubland_field_delineation


@api_view(["POST"])
def generate_farm_boundary(request):
    print("Inside generate_farm_boundary API.")
    try:
        state = request.data.get("state").lower()
        district = request.data.get("district").lower()
        block = request.data.get("block").lower()
        scrubland_field_delineation.apply_async(
            args=[state, district, block], queue="core_stack"
        )
        return Response(
            {"Success": "Successfully initiated"}, status=status.HTTP_200_OK
        )
    except Exception as e:
        print("Exception in generate_farm_boundary api :: ", e)
        return Response({"Exception": e}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
