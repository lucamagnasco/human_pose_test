# editting saved datapoints
import math

# tennis example: https://colab.research.google.com/drive/1Rf1abfT8yUOEm73glwjUS2-FKH86syFK

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return ang


def computeElbowAngle(row, which='right'):
    wrist = Point(row[f'{which}_wrist_x'], row[f'{which}_wrist_y'])
    elbow = Point(row[f'{which}_elbow_x'], row[f'{which}_elbow_y'])
    shoulder = Point(row[f'{which}_shoulder_x'], row[f'{which}_shoulder_y'])
    return getAngle(wrist, elbow, shoulder)


def computeShoulderAngle(row, which='right'):
    elbow = Point(row[f'{which}_elbow_x'], row[f'{which}_elbow_y'])
    shoulder = Point(row[f'{which}_shoulder_x'], row[f'{which}_shoulder_y'])
    hip = Point(row[f'{which}_hip_x'], row[f'{which}_hip_y'])
    return getAngle(hip, shoulder, elbow)


def computeKneeAngle(row, which='right'):
    hip = Point(row[f'{which}_hip_x'], row[f'{which}_hip_y'])
    knee = Point(row[f'{which}_knee_x'], row[f'{which}_knee_y'])
    ankle = Point(row[f'{which}_ankle_x'], row[f'{which}_ankle_y'])
    return getAngle(ankle, knee, hip)