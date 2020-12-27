# -*- coding: UTF-8 -*-

config={
    "data_dir": "data",
    "data_file": "data/Geolife_total.txt",
    "middle_size_data_file": "data/middle_size_data.txt",
    "small_size_data_file": "data/small_size_data.txt",

    "area_num": 20,

    "related_day_num": 3,   # how many day's data that is related to today
    "related_week_num": 3,  # how many week's data that is related to today
    "related_hour_num": 3,  # how many hour's data that is related to today

    "train_validate_ratio": "7:3",  # train example vs validate example  ratio

}


print(config)