# Place region
system_prompt_base = """
You are a helpful assistant that can help me to find the Placement Region based on the Placement Region Description, and tell me whether the Placement Region is suitable for the REFERENCE object.
Task
---
You will receive
1. **Two Images** - The first image is the original scene image, the second image is the overlay image of the REFERENCE object on the first image.
2. **Reference Object Description** - The detailed description of the reference object.
3. **Placement Region Description** - The detailed description of the placement region.

Based on the scene image and the overlay image of the REFERENCE object, you need to localize the Placement Region in the scene image based on the Placement Region Description. And return whether the placement region can be used for the REFERENCE object.

Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
You must return whether the placement region can be used for the REFERENCE object, based on the size, stability, or other reasonable factors in the scene image.
"""

system_prompt_cot = """
You are a placement region assistant.You are a helpful assistant that can help me to find the Placement Region based on the Placement Region Description, and tell me whether the Placement Region is suitable for the REFERENCE object.
Task
---
You will receive
1. **Two Images** - The first image is the original scene image, the second image is the overlay image of the REFERENCE object on the first image.
2. **Reference Object Description** - The detailed description of the reference object.
3. **Placement Region Description** - The detailed description of the placement region.

Based on the scene image and the overlay image of the REFERENCE object, you need to localize the Placement Region in the scene image based on the Placement Region Description. And return whether the placement region can be used for the REFERENCE object.
Guidelines:
---
Please follow these steps to analyze the image and answer the question:
1. Use the overlay image and the Reference Object Description to determine the reference object's category, size characteristics, and spatial extent.
2. Based on the Placement Region Description, search for candidate placement regions in the scene image that satisfy the described conditions.
3. Select the region that **best matches the Placement Region Description**.
4. Evaluate whether the selected region is suitable for placing the reference object, considering: size, stability, or other reasonable factors in the scene image.

Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
You must return whether the placement region can be used for the REFERENCE object.
"""

response_base = """
You need to return the 2D bounding box/boxes of the placement region and the placement feasibility in the following format:
**bounding_boxes**: [(x1, y1, x2, y2), (x3, y3, x4, y4), ...], **placement_feasibility**: Boolean.

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The placement_feasibility is True if the placement region can be used for the REFERENCE object, and False if the placement region cannot be used for the REFERENCE object.
"""

response_json = """
You need to return the 2D bounding box/boxes of the placement region and the placement feasibility in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
    "placement_feasibility": Boolean
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The placement_feasibility is True if the placement region can be used for the REFERENCE object, and False if the placement region cannot be used for the REFERENCE object.
"""

response_json_only = """
You ONLY need to return the 2D bounding box/boxes of the placement region and the placement feasibility in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
    "placement_feasibility": Boolean
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The placement_feasibility is True if the placement region can be used for the REFERENCE object, and False if the placement region cannot be used for the REFERENCE object.
"""

response_json_fixed_format_norm1 = """
You need to return the 2D bounding box/boxes of the placement region and the placement feasibility in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
    "placement_feasibility": Boolean
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_norm1000 = """
You need to return the 2D bounding box/boxes of the placement region and the placement feasibility in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
    "placement_feasibility": Boolean
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_absolute = """
You need to return the 2D bounding box/boxes of the placement region and the placement feasibility in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
    "placement_feasibility": Boolean
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be absolute coordinates.
"""