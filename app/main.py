from model import Data
from views import head, body


def main():
    data = Data()
    head()
    body(data)


main()
