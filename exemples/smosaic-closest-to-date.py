import os
from smosaic import mosaic

stac_url = "https://data.inpe.br/bdc/stac/v1"

result = mosaic(
    name="luis-eduardo-magalhaes",
    data_dir=os.path.abspath(""),
    stac_url=stac_url,
    collection="S2_L2A-1", 
    bbox="-47.8530,-5.0246,-47.5399,-4.7578",
    output_dir=os.path.join("output"),   
    mosaic_method="ctd", 
    start_year=2025,
    start_month=9,
    start_day=1,
    reference_date="2025-09-15",
    duration_months=1, 
    bands=["B02","B03","B04"]
)

