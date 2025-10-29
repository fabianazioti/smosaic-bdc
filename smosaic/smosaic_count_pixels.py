import rasterio


def count_pixels(raster_path, target_value):
    """
    Counts the number of pixels in a raster that match a specific value.
    
    Args:
        raster_path (str): Path to the raster file
        target_value (int/float): The pixel value to count
        
    Returns:
        int: Count of pixels with the target value
    """
    
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        
        count = (data == target_value).sum()
        
        return dict(total=data.size, count=count)
