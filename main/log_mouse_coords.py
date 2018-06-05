"""
Log your mouse coordinates to the screen.
"""
import time
import win32api


def log_mouse_coords(sleep_time=0.5, intervals=30):
    """
        Log the mouse position for a certain amount of time.
    """
    for interval in range(intervals):
        time.sleep(sleep_time)
        x_pos, y_pos = win32api.GetCursorPos()
        print(interval)
        print("X coord:".ljust(10) + str(x_pos).ljust(5))
        print("Y coord:".ljust(10) + str(y_pos).ljust(5) + "\n")
    return


if __name__ == "__main__":
    log_mouse_coords()
