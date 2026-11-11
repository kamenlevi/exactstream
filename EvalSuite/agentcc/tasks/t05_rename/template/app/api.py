from app.users import get_user


def handle(uid):
    return {'user': get_user(uid)}
