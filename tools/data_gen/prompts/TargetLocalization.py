## Target localization
system_prompt_base = """
You are a target localization assistant. Your job is to localize the target objects in the image based on the target object description.

Task
---
You will receive
1. **Image** - An RGB image of the scene.
2. **Target Object Description** - The detailed description of the target object to be localized.

Based on the image and the target object description, you need to localize all the target objects in the image. Notice, there might be multiple or none target objects in the image. 

The number of the objects in your answer should be accurate and consistent with the target object description.
Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

system_prompt_cot = """
You are a target localization assistant. Your job is to localize the target objects in the image based on the target object description.

Task
---
You will receive
1. **Image** - An RGB image of the scene.
2. **Target Object Description** - The detailed description of the target object to be localized.

Based on the image and the target object description, you need to localize all the target objects in the image. Notice, there might be multiple or none target objects in the image. 
Guidelines:
---
Please follow these steps to analyze the image and answer the question:
1. Interpret the Target Object Description to identify key attributes (category, properties, appearance cues).
2. Search the image for candidate objects that may satisfy the description.
3. Select objects whose visual characteristics are consistent with the description.
4. Verify the number of matched target objects for consistency with the scene and description.

Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

response_base = """
You need to return the 2D bounding box/boxes of the target object and the number of the target object in the image in the following format:
**bounding_boxes**: [(x1, y1, x2, y2), (x3, y3, x4, y4), ...], **number_of_objects**: int.

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
""" 

response_json ="""
You need to return the the number of the target object and the 2D bounding box/boxes of the target objects in the image in the following JSON format:
{
    "number_of_objects": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],    

}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_only = """
You ONLY need to return the the number of the target object and the 2D bounding box/boxes of the target objects in the image in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "number_of_objects": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],    
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_fixed_format_norm1 = """
You need to return the the number of the target object and the 2D bounding box/boxes of the target objects in the image in the following JSON format:
{
    "number_of_objects": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],    
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_norm1000 = """
You need to return the the number of the target object and the 2D bounding box/boxes of the target objects in the image in the following JSON format:
{
    "number_of_objects": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],    
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_absolute = """
You need to return the the number of the target object and the 2D bounding box/boxes of the target objects in the image in the following JSON format:
{
    "number_of_objects": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],    
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be absolute coordinates.
"""