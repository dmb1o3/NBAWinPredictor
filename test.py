from data_collector import is_back_to_back, handle_year_input
from pandas import Series


def test_handle_year_input():
    print("Testing handle_year_input() from data_collector.py")
    assert handle_year_input("2023") == ["2023"]
    assert handle_year_input("2020 2021 2022 2023") == ["2020", "2021", "2022", "2023"]
    assert handle_year_input("2020-2023") == ["2020", "2021", "2022", "2023"]
    assert handle_year_input("2018 2020-2023") == ["2018", "2020", "2021", "2022", "2023"]
    assert handle_year_input("2020-2023 2024") == ["2020", "2021", "2022", "2023", "2024"]
    print("Everything Passed\n")


def test_is_back_to_back():
    print("Testing is_back_to_back() from data_collector.py")
    # Test sequential
    assert is_back_to_back(Series("2022-11-02"), Series("2022-11-03")) == 1
    # Test sequential but different month
    assert is_back_to_back(Series("2022-10-02"), Series("2022-11-03")) == 0
    # Test sequential but different year. Should never happen but better to classify properly
    assert is_back_to_back(Series("2021-11-02"), Series("2022-11-03")) == 0
    # Test february
    assert is_back_to_back(Series("2022-02-02"), Series("2022-02-03")) == 1
    assert is_back_to_back(Series("2024-02-27"), Series("2024-02-28")) == 1
    assert is_back_to_back(Series("2024-02-28"), Series("2024-02-29")) == 1
    assert is_back_to_back(Series("2020-02-28"), Series("2020-02-29")) == 1
    assert is_back_to_back(Series("2016-02-28"), Series("2016-02-29")) == 1
    assert is_back_to_back(Series("2023-02-28"), Series("2023-02-29")) == 0
    assert is_back_to_back(Series("2024-02-28"), Series("2025-02-29")) == 0
    assert is_back_to_back(Series("2023-02-27"), Series("2023-02-28")) == 1
    assert is_back_to_back(Series("2020-02-28"), Series("2020-02-29")) == 1
    assert is_back_to_back(Series("2019-02-28"), Series("2020-02-29")) == 0
    # Test december
    assert is_back_to_back(Series("2019-12-31"), Series("2020-01-1")) == 1
    assert is_back_to_back(Series("2019-12-31"), Series("2020-01-2")) == 0
    assert is_back_to_back(Series("2019-12-30"), Series("2020-01-1")) == 0
    assert is_back_to_back(Series("2018-12-31"), Series("2020-01-1")) == 0
    # Test for 30 month
    assert is_back_to_back(Series("2022-04-30"), Series("2022-05-01")) == 1
    assert is_back_to_back(Series("2021-06-30"), Series("2021-07-01")) == 1
    assert is_back_to_back(Series("2021-06-30"), Series("2021-07-01")) == 1
    assert is_back_to_back(Series("2022-04-31"), Series("2022-05-01")) == 0
    assert is_back_to_back(Series("2021-06-31"), Series("2021-07-01")) == 0
    assert is_back_to_back(Series("2021-06-31"), Series("2021-07-01")) == 0
    # Test for 31 month
    assert is_back_to_back(Series("2022-01-31"), Series("2022-02-01")) == 1
    assert is_back_to_back(Series("2021-05-31"), Series("2021-06-01")) == 1
    assert is_back_to_back(Series("2011-05-31"), Series("2011-06-01")) == 1
    assert is_back_to_back(Series("2022-01-30"), Series("2022-02-01")) == 0
    assert is_back_to_back(Series("2021-05-30"), Series("2021-06-01")) == 0
    assert is_back_to_back(Series("2011-05-30"), Series("2011-06-01")) == 0

    # Test roll over to next month
    assert is_back_to_back(Series("2022-01-31"), Series("2022-02-01")) == 1
    assert is_back_to_back(Series("2022-01-31"), Series("2022-02-01")) == 1
    assert is_back_to_back(Series("2022-01-31"), Series("2022-02-01")) == 1
    assert is_back_to_back(Series("2022-01-31"), Series("2022-02-01")) == 1
    assert is_back_to_back(Series("2022-01-31"), Series("2022-02-01")) == 1

    # Test rollover for february
    assert is_back_to_back(Series("2020-02-29"), Series("2020-03-01")) == 1
    assert is_back_to_back(Series("1996-02-29"), Series("1996-03-01")) == 1
    assert is_back_to_back(Series("2022-02-28"), Series("2022-03-01")) == 1

    print("Everything passed\n")


def main():
    test_handle_year_input()
    test_is_back_to_back()


if __name__ == "__main__":
    main()