# feasible path (the first perspective)
system_prompt_first_perspective = """
You are a helpful assistant that can help me to find the feasible path to reach the target area.
Task
---
You will receive
1. **Two Images** - The first image is the original image from the first perspective, the second image is the overlay image of the target area on the first image.
2. **Target Area Description** - The detailed description of the target area.

Based on the scene image and the overlay image of the target area and the "target area description", you need to find the feasible path to reach the target area in the scene image. 

The path should start from the place where the picture/photo is taken.
The path you find should be the shortest path, but the path in your answer must all be on the ground area.
The path you find must be a path that exists, though it might be visually blocked. Visually blocked means the path you find may not be seen from the image with the current perspective, but the actual feasible path exists.
"""

system_prompt_first_perspective_cot = """
You are a helpful assistant that can help me to find the feasible path to reach the target area.
Task
---
You will receive
1. **Two Images** - The first image is the original image from the first perspective, the second image is the overlay image of the target area on the first image.
2. **Target Area Description** - The detailed description of the target area.

Based on the scene image and the overlay image of the target area and the "target area description", you need to find the feasible path to reach the target area in the scene image. 
Guidelines:
---
Please follow these steps to analyze the image and answer the question:
1. Use the overlay image and the "target area description" to precisely determine the target area's location and extent in the original image.
2. Identify all **traversable supporting surfaces**, including: Flat ground (floor, road), Ramps/slopes, Stairs. Distinguish them from non-traversable regions (e.g.walls, tabletops, obstacles, air/elevated void regions).
3. Within the traversable supporting surfaces only, plan a continuous feasible path from the start position to the target area. If multiple feasible paths exist, choose the shortest one under the ground constraint.

The path should start from the place where the picture/photo is taken.
The path you find must be a path that exists, though it might be visually blocked. Visually blocked means the path you find may not be seen from the image with the current perspective, but the actual feasible path exists.
"""

response_base_1 = """
You need to return a list of the points to represent the path in the following format:
**Points**: [(x1, y1), (x2, y2),(x3, y3), ...]

Each point is in the format of (x, y), where x is the x coordinate of the point, and y is the y coordinate of the point.
"""

response_json_1 = """
You need to return a list of points to represent the path in the following JSON format:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
""" 

response_json_only_1 = """
You ONLY need to return a list of points to represent the path in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}
where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
"""

response_json_fixed_format_1_norm1 = """
You need to return a list of points to represent the path in the following JSON format:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_1_norm1000 = """
You need to return a list of points to represent the path in the following JSON format:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_1_absolute = """
You need to return a list of points to represent the path in the following JSON format:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
The output coordinates MUST be absolute coordinates.
"""


##############################
## the third perspective
##############################

system_prompt_third_perspective = """
You are a helpful assistant that can help me to find the feasible path from one target area to another target area.
Task
---
You will receive
1. **Two Images** - The first image is the original image from the third perspective, the second image is the overlay image of all the target areas on the first image.
2. **Target Area Description** - The detailed description of the target areas.

Based on the scene image and the overlay image of the target areas and the "target area description", you need to find the feasible path from one target area to another target area in the scene image. 

The path you find must be a path that exists, though it might be visually blocked. Visually blocked means the path you find may not be seen from the image with the current perspective, but the actual feasible path exists.
The path you find should be the shortest path, but the path in your answer must all be on the ground area.
"""

system_prompt_third_perspective_cot = """
You are a helpful assistant that can help me to find the feasible path from one target area to another target area.
Task
---
You will receive
1. **Two Images** - The first image is the original image from the third perspective, the second image is the overlay image of all the target areas on the first image.
2. **Target Area Description** - The detailed description of the target areas.

Based on the scene image and the overlay image of the target areas and the "target area description", you need to find the feasible path from one target area to another target area in the first image.

Guidelines:
---
Please follow these steps to analyze the image and answer the question:
1. Use the overlay image to precisely determine the target area's location and extent in the original image.
2. Identify all **traversable supporting surfaces**, including: Flat ground (floor, road), Ramps/slopes, Stairs. Distinguish them from non-traversable regions (e.g.walls, tabletops, obstacles, air/elevated void regions).
3. Within the traversable supporting surfaces only, plan a continuous feasible path from one target area to another target area. If multiple feasible paths exist, choose the shortest one under the ground constraint.

The path you find must be a path that exists, though it might be visually blocked. Visually blocked means the path you find may not be seen from the image with the current perspective, but the actual feasible path exists.
"""


response_base_3 = """
You need to return a list of the points to represent the path from one target area to another target area in the following format:
**Points**: [(x1, y1), (x2, y2), ...]

Each point is in the format of (x, y), where x is the x coordinate of the point, and y is the y coordinate of the point.
"""

response_json_3 = """
You need to return a list of multiple points to represent the path from one target area to another target area in the following JSON format:    
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
"""

response_json_only_3 = """
You ONLY need to return a list of multiple points to represent the path from one target area to another target area in the following JSON format WITHOUT ANY OTHER TEXT:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}
where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
"""

response_json_fixed_format_3_norm1 = """
You need to return a list of multiple points to represent the path from one target area to another target area in the following JSON format:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
The output coordinates MUST be normalized to a 0-1 scale.
"""

response_json_fixed_format_3_norm1000 = """
You need to return a list of multiple points to represent the path from one target area to another target area in the following JSON format:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
The output coordinates MUST be normalized to a 0-1000 scale.
"""

response_json_fixed_format_3_absolute = """
You need to return a list of multiple points to represent the path from one target area to another target area in the following JSON format:
{
    "points": [(x1, y1), (x2, y2),(x3, y3),...,(xn, yn)]
}

where x is the x coordinate of the point, y is the y coordinate of the point, and n >= 3 is the number of points in the path.
The output coordinates MUST be absolute coordinates.
"""