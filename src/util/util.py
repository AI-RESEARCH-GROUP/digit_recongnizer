# -*- coding: UTF-8 -*-

import os
import datetime



def compute_root_dir():
    root_dir = os.path.abspath(
        os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__))))
    return root_dir + os.path.sep


proj_root_dir = compute_root_dir()


def file_exists(file_path):
    return os.path.exists(file_path)


def get_date_before(date_str, num):
    result = []
    for n in range(num, 0, -1):
        n_days_ago = datetime.datetime.strptime(date_str, "%Y-%m-%d") - datetime.timedelta(n)
        result.append(n_days_ago.strftime("%Y-%m-%d"))
    return result


def get_week_before(date_str, num):
    result = []
    for n in range(num, 0, -1):
        n_days_ago = datetime.datetime.strptime(date_str, "%Y-%m-%d") - datetime.timedelta(n * 7)
        result.append(n_days_ago.strftime("%Y-%m-%d"))
    return result


if __name__ == "__main__":
    # print(proj_root_dir)
    print(get_week_before("2020-08-02", 3))