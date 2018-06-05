"""
You can use this function to add data to your WSR.
"""
import datetime
import openpyxl


def add_data_to_wsr():
    """Add the work I did to my WSR."""
    file = openpyxl.load_workbook('PAUL-WSR.xlsx')
    sheet = file.get_sheet_by_name('Paul')
    task_name = input("What is the name of the task?")
    desc = input("What is the description of the task?")
    owner = input("Who assigned you this task?")
    hours = input("How many hours did you spend on this task?")
    project = input("What project is this for?")
    num = str(sheet.max_row + 1)
    sheet['A' + num] = task_name
    sheet['B' + num] = desc
    sheet['C' + num] = owner.upper()
    sheet['D' + num] = hours
    sheet['E' + num] = datetime.datetime.now().strftime('%m/%d/%Y')
    sheet['F' + num] = project.upper()
    file.save('PAUL-WSR.xlsx')
    return


if __name__ == "__main__":
    add_data_to_wsr()
    print("Added!")
