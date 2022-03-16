from enum import Enum
from urllib.request import UnknownHandler


class Computer(Enum):
    UHA     = 0
    MESOCTR = 1
    CASIMIR = 2

curr_computer = Computer.CASIMIR

if curr_computer == Computer.MESOCTR:
    # Path to datasets
    PATH_UCR_ARCHIVE_2015 = "/home2020/home/uha/gpialla/datasets/UCRArchive_2015"
    PATH_UCR_ARCHIVE_2018 = "/home2020/home/uha/gpialla/datasets/UCRArchive_2018"

    # Default output directory
    DEFAULT_OUTPUT_DIR = "/home2020/home/uha/gpialla/results"

elif curr_computer == Computer.CASIMIR:
    # Path to datasets
    PATH_UCR_ARCHIVE_2015 = "/home/pialla/Documents/UCRArchive_2015"
    PATH_UCR_ARCHIVE_2018 = "/home/pialla/Documents/UCRArchive_2018"

    # Default output directory
    DEFAULT_OUTPUT_DIR = "/home/pialla/Documents/Results"