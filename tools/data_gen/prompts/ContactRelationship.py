## ContactRelationship class 1
system_prompt_base_1 = """
You are a helpful assistant that can help me to localize the TARGET objects that are DIRECTLY in contact with the REFERENCE object.
Task    
---
You will receive
1. **Two Images** - The first one is the image of the scene, the second one is the overlay image of the REFERENCE object on the scene image.
2. **Reference Object Description** - The detailed description of the reference object.

Based on the scene image and the overlay image of the REFERENCE object and the reference object description, you need to find all the TARGET objects that are DIRECTLY in contact with the REFERENCE object.

You must return the number of TARGET objects that are DIRECTLY in contact with the REFERENCE object.
The TARGET objects you return must NOT include any object that is excessively larger than the REFERENCE object.
Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

system_prompt_cot_1 = """
You are a helpful assistant that can help me to localize the TARGET objects that are DIRECTLY in contact with the REFERENCE object.
Task
---
You will receive
1. **Two Images** - The first one is the image of the scene, the second one is the overlay image of the REFERENCE object on the scene image.
2. **Reference Object Description** - The detailed description of the reference object.
Based on the scene image and the overlay image of the REFERENCE object and the reference object description, you need to find all the TARGET objects that are DIRECTLY in contact with the REFERENCE object.

Guidelines:
---
Please follow these steps to analyze the image and answer the question:
1. Use the second overlay image and the "reference object description" to accurately localize the REFERENCE object in the scene image, and determine its category and spatial extent.
2. Identify all candidate TARGET objects that are in **direct physical contact** with the reference object.
3. Based on visual evidence and common-sense reasoning, exclude objects that are clearly unlikely to be in direct contact with the reference object.
4. Ensure the TARGET object is truly in direct contact, not merely close to or visually overlapping.

You must return the number of TARGET objects that are DIRECTLY in contact with the REFERENCE object.
The TARGET objects you return must NOT include any object that is excessively larger than the REFERENCE object.
Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

response_base_1 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with the REFERENCE object and the 2D bounding box/boxes of the TARGET objects in the following format:
**number_of_instances**: int, **bounding_boxes**: [(x1, y1, x2, y2), (x3, y3, x4, y4), ...].

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_1 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with the REFERENCE object and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}   
Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_only_1 = """
You ONLY need to return the number of the TARGET objects that are DIRECTLY in contact with the REFERENCE object and the 2D bounding box/boxes of the TARGET objects in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}
Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_fixed_format_1_norm1 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with the REFERENCE object and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_1_norm1000 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with the REFERENCE object and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_1_absolute = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with the REFERENCE object and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be absolute coordinates.
"""

## ContactRelationship class 2
system_prompt_base_2 = """
You are a helpful assistant that can help me to localize the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects.
Task
---
You will receive
1. **Two Images** - The first one is the image of the scene, the second one is the overlay image of at least two REFERENCE objects on the scene image.
2. **Reference Object Description** - The detailed description of the reference objects.

Based on the scene image and the overlay image of the REFERENCE objects and the reference object description, you need to localize all the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects.

You must return the number of the objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL target objects.
The TARGET objects you return must NOT include any object that is excessively larger than the REFERENCE objects.
Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

system_prompt_cot_2 = """
You are a helpful assistant that can help me to localize the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects.
Task
---
You will receive
1. **Two Images** - The first one is the image of the scene, the second one is the overlay image of at least two REFERENCE objects on the scene image.
2. **Reference Object Description** - The detailed description of the reference objects.

Based on the scene image and the overlay image of the REFERENCE objects and the reference object description, you need to localize all the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects.

Guidelines
---
Please follow these steps to analyze the image and answer the question:
1. Use the overlay image and the "reference object description" to accurately localize **all** REFERENCE objects in the scene image and determine their spatial extents.
2. Identify candidate TARGET objects that are in **direct physical contact** with any reference object.
3. Select only those TARGET objects that are **directly and simultaneously in contact with all REFERENCE objects**
4. Based on visual evidence and common-sense reasoning, exclude objects that are clearly unlikely to be in direct contact with ALL REFERENCE objects.
5. Ensure the TARGET object is truly in direct contact, not merely close to or visually overlapping.

You must return the number of the objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL target objects.
The TARGET objects you return must NOT include any object that is excessively larger than the REFERENCE objects.
Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

response_base_2 = """
You need to return the number of the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following format:
**number_of_instances**: int, **bounding_boxes**: [(x1, y1, x2, y2), (x3, y3, x4, y4), ...].

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_2 = """
You need to return the number of the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_only_2 = """
You ONLY need to return the number of the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}
Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_fixed_format_2_norm1 = """
You need to return the number of the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_2_norm1000 = """
You need to return the number of the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_2_absolute = """
You need to return the number of the TARGET objects that are DIRECTLY and SIMULTANEOUSLY in contact with ALL REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be absolute coordinates.
"""

## ContactRelationship class 3
system_prompt_base_3 = """
You are a helpful assistant that can help me to localize the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects.
Task
---
You will receive
1. **Two Images** - The first one is the image of the scene, the second one is the overlay image of at least two REFERENCE objects on the scene image.
2. **Reference Object Description** - The detailed description of the reference objects.

Based on the scene image and the overlay image of the REFERENCE objects and the "reference object description", you need to localize all the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects.

You must return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects.
The TARGET objects you return must NOT include any object that is excessively larger than the REFERENCE objects.
Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

system_prompt_cot_3 = """
You are a helpful assistant that can help me to localize the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects.
Task
---
You will receive
1. **Two Images** - The first one is the image of the scene, the second one is the overlay image of at least two REFERENCE objects on the scene image.
2. **Reference Object Description** - The detailed description of the reference objects.

Based on the scene image and the overlay image of the REFERENCE objects and the "reference object description", you need to localize all the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects.

Guidelines
---
Please follow these steps to analyze the image and answer the question:
1. Use the overlay image and the "reference object description" to accurately localize all REFERENCE objects and determine their spatial extents.
2. Identify candidate TARGET objects that are in direct physical contact with any reference object.
3. Based on visual evidence and basic common sense, exclude objects that clearly violate physical contact relationships.
4. Ensure the TARGET object is truly in direct contact, not merely close to or visually overlapping.

You must return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects.
The TARGET objects you return must NOT include any object that is excessively larger than the REFERENCE objects.
Your localization answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

response_base_3 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following format:
**number_of_instances**: int, **bounding_boxes**: [(x1, y1, x2, y2), (x3, y3, x4, y4), ...].

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_3 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_only_3 = """
You ONLY need to return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}
Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
"""

response_json_fixed_format_3_norm1 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_3_norm1000 = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_3_absolute = """
You need to return the number of the TARGET objects that are DIRECTLY in contact with ANY REFERENCE objects and the 2D bounding box/boxes of the TARGET objects in the following JSON format:
{
    "number_of_instances": int,
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...],
}

Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner of the bounding box.
The output coordinates MUST be absolute coordinates.
"""