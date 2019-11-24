from removebg import RemoveBg
import os
import sys
import shutil

rmbg = RemoveBg("CYWAS2zeFoVQGVFHESJ8LrJU", "error.log")
for i in range(1,18):
    rmbg.remove_background_from_img_file('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_human/human_%04d.jpg'%(i))
    print("fs")

for i in range(1, 18):
    if os.path.exists('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_human/human_%04d.jpg_no_bg.png'%(i)):
        shutil.move('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_human/human_%04d.jpg_no_bg.png'%(i),
        '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_human_crop')
    else:
        shutil.move('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_human/human_%04d.jpg'%(i),
                    '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_human_crop')