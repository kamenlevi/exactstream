from app.users import fetch_user


def handle(uid):
    return {'user': fetch_user(uid)}
