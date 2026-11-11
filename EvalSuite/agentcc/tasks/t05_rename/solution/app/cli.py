from app import users


def main(uid):
    print(users.fetch_user(uid))
