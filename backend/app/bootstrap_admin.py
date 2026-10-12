"""Create the first Slidronix administrator using the configured database."""

import getpass
import sys

from sqlalchemy import select

from app.auth import Role, User, password_hash
from app.database import SessionLocal


def main() -> int:
    email = input("Administrator email: ").strip().lower()
    name = input("Administrator name: ").strip()
    password = getpass.getpass("Password (at least 12 characters): ")
    if len(password) < 12:
        print("Password must contain at least 12 characters.", file=sys.stderr)
        return 2

    with SessionLocal() as db:
        if db.scalar(select(User.id).where(User.role == Role.admin.value).limit(1)):
            print("An administrator already exists; bootstrap is disabled.", file=sys.stderr)
            return 1
        if db.scalar(select(User.id).where(User.email == email)):
            print("That email already belongs to an account.", file=sys.stderr)
            return 1
        account = User(name=name, email=email, password_hash=password_hash.hash(password), role=Role.admin.value)
        db.add(account)
        db.commit()
    print("Administrator account created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
