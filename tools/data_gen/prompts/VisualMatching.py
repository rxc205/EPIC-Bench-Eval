## Visual matching
system_prompt_base = """
You are a helpful assistant that can help me to match the areas/objects in two different perspectives.
Task
--- 
You will receive
1. **Three Images** - The first image is the original scene from one perspective, the second image is the overlay image of the REFERENCE areas/objects in the first image, and the third is the image of the original scene from a different perspective.

Based on the first and second images, you need to locate the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images.

Your matching answer of the TARGET areas/objects must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth TARGET Areas/Objects as much as possible.
"""

system_prompt_cot = """
You are a helpful assistant that can help me to match the areas/objects in two different perspectives.
Task
--- 
You will receive
1. **Three Images** - The first image is the original scene from one perspective, the second image is the overlay image of the REFERENCE areas/objects in the first image, and the third is the image of the original scene from a different perspective.

Based on the first and second images, you need to locate the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images.

Guidelines:
---
Please follow these steps to analyze the image and answer the question:
1. Use the first and second images to precisely determine the location and visual characteristics of the REFERENCE areas/objects.
2. Interpret the viewpoint differences between the first and third images (e.g., translation, scale change, rotation, occlusion).
3. Search for candidate TARGET areas/objects in the third image using visual cues such as appearance, geometry, and spatial relationships.
4. Select the TARGET areas/objects that are most consistent with the REFERENCE areas/objects.

Your matching answer of the TARGET areas/objects must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth TARGET Areas/Objects as much as possible.
"""

response_base = """
You need to return the number of the TARGET areas/objects and the 2D bounding box/boxes of the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images in the following format:
**bounding_boxes**: [(x1, y1, x2, y2), (x3, y3, x4, y4), ...].

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json = """
You need to return the number of the TARGET areas/objects and the 2D bounding box/boxes of the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""
response_json_only = """
You ONLY need to return the number of the TARGET areas/objects and the 2D bounding box/boxes of the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""
response_json_fixed_format_norm1 = """
You need to return the number of the TARGET areas/objects and the 2D bounding box/boxes of the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1 scale.
"""
response_json_fixed_format_norm1000 = """
You need to return the number of the TARGET areas/objects and the 2D bounding box/boxes of the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_absolute = """
You need to return the number of the TARGET areas/objects and the 2D bounding box/boxes of the TARGET areas/objects in the third image that match the REFERENCE areas/objects in the first and second images in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be absolute coordinates.
"""