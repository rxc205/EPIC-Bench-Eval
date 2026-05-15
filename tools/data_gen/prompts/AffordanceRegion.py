# Affordance
system_prompt_base = """
You are a helpful assistant that can help me to find the Affordance Region of the Reference Object in the image based on the Task Description.
Task
---
You will receive
1. **Two Images** - The first image is the original image, the second image is the overlay image of the REFERENCE object on the first image.
2. **Reference Object Description** - The detailed description of the reference object.
3. **Task Description** - The detailed description of the task description.

You need to localize the Affordance Region of the REFERENCE object that can be used to complete the task in the first image.

Your answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""

system_prompt_cot = """
You are a helpful assistant that can help me to find the Affordance Region of the Reference Object in the image based on the Task Description.
Task
---
You will receive
1. **Two Images** - The first image is the original image, the second image is the overlay image of the REFERENCE object on the first image.
2. **Reference Object Description** - The detailed description of the reference object.
3. **Task Description** - The detailed description of the task description.

You need to localize the Affordance Region of the REFERENCE object that can be used to complete the task in the first image.
Guidelines:
---
Please follow these steps to analyze the image and answer the question:
1. Use Image B (overlay) to precisely locate the reference object within Image A, and verify its consistency with the "Reference Object Description".
2. Analyze the "Task Description" to determine the type of interaction required for the task (such as grabbing, pressing, pulling, etc.).
3. Based on the type of interaction required for the task, localize the Affordance Region of the REFERENCE object that can be used to complete the task in the first image.

Your answer must be precise, and try to maximize the Intersection over Union (IOU) with the Ground Truth as much as possible.
"""



response_base = """
You need to return the 2D bounding box/boxes of the Affordance Region of the REFERENCE object that is used to complete the task in the following format:
**bounding_boxes**: [(x1, y1, x2, y2), (x3, y3, x4, y4), ...].

Each 2D bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the 2D bounding box, and (x2, y2) is the bottom-right corner of the 2D bounding box.
"""

response_json = """
You need to return the 2D bounding box/boxes of the Affordance Region of the REFERENCE object that is used to complete the task in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each 2D bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the 2D bounding box, and (x2, y2) is the bottom-right corner of the 2D bounding box.
"""

response_json_only = """
You ONLY need to return the 2D bounding box/boxes of the Affordance Region of the REFERENCE object that is used to complete the task in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each 2D bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the 2D bounding box, and (x2, y2) is the bottom-right corner of the 2D bounding box.
"""

response_json_fixed_format_norm1 = """
You need to return the 2D bounding box/boxes of the Affordance Region of the REFERENCE object that is used to complete the task in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each 2D bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the 2D bounding box, and (x2, y2) is the bottom-right corner of the 2D bounding box.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_norm1000 = """
You need to return the 2D bounding box/boxes of the Affordance Region of the REFERENCE object that is used to complete the task in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each 2D bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the 2D bounding box, and (x2, y2) is the bottom-right corner of the 2D bounding box.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_absolute = """
You need to return the 2D bounding box/boxes of the Affordance Region of the REFERENCE object that is used to complete the task in the following JSON format:
{
    "bounding_boxes": [(x1, y1, x2, y2), (x3, y3, x4, y4), ...]
}

Each 2D bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the 2D bounding box, and (x2, y2) is the bottom-right corner of the 2D bounding box.
The output coordinates MUST be absolute coordinates.
"""