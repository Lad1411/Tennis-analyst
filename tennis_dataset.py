import json
import os
import shutil

folder_path = r"D:\Python Files\AI\Deeplearning\Tennis\Dataset\tennis_ball\valid"
destination_path = r"D:\Python Files\AI\Deeplearning\Tennis\Dataset\labels"

image_set = "validation"

anot_file = os.path.join(folder_path, "_annotations.coco.json")

with open(anot_file) as json_file:
    f = json.load(json_file)

start = 0
last_idx = 2575

for data in f["images"]:
    img_name = data["file_name"]
    img_id = data["id"]
    img_width = data["width"]
    img_height = data["height"]
    # last_id = img_id


    # prev_path = os.path.join(folder_path, img_name)
    # new_path = os.path.join(destination_path, image_set+ r"\img{}.jpg".format(img_id+last_idx))
    #
    # shutil.copy2(prev_path, new_path)

    txt_filename = "img{}.txt".format(img_id+last_idx)
    folder_name = os.path.join(destination_path, image_set)
    txt_file = os.path.join(folder_name, txt_filename)

    file = open(txt_file, "a")

    for idx in range(start, len(f["annotations"])):
        if f["annotations"][idx]["image_id"] > img_id:
            break

        elif f["annotations"][idx]["image_id"] ==  img_id:
            properties = f["annotations"][idx]
            start = idx+1
            category_id = properties["category_id"]
            x,y,width,height = properties["bbox"]

            xcenter = x+width/2
            ycenter = y+height/2

            xcenter = xcenter/img_width
            width = width/img_width

            ycenter = ycenter/img_height
            height = height/img_height

            file.write("{} {} {} {} {}\n".format(category_id, xcenter, ycenter, width, height))


    file.close()








