#!/usr/bin/env python

import pathlib
import csv
import time

import structlog
import click

from awsauth import auth_app
from awsauth import ses_client
from awsauth.email_repo import EmailRepo


_logger = structlog.get_logger()


RISK_LEVEL_UPDATE_PATH = pathlib.Path(__file__).parent / "nyc_2_0_announcement.html"


def _load_emails(path: pathlib.Path):

    with path.open() as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row["email"]


def _build_email(to_email: str, email_html_path: pathlib.Path) -> ses_client.EmailData:
    email_html = email_html_path.read_text()

    return ses_client.EmailData(
        subject="Change in NYC Data Aggregation",
        from_email="CovidActNow API <api@covidactnow.org>",
        reply_to="api@covidactnow.org",
        to_email=to_email,
        html=email_html,
        configuration_set="api-risk-level-update-emails",
    )


@click.command()
@click.argument("email-list-path", type=pathlib.Path)
@click.argument("email-path", type=pathlib.Path)
def send_emails(
    email_list_path: pathlib.Path, email_path: pathlib.Path
):  # pylint: disable=no-value-for-parameter

    auth_app.init()

    emails = _load_emails(email_list_path)

    for email in emails:
        print(email)
        risk_email = _build_email(email, email_path)

        email_send_result = EmailRepo.send_email(risk_email)
        if not email_send_result:
            _logger.warning(f"Failed to send email to {email}")

        # SES rate limit is 14 messages a second, pause after email sends to not
        # trigger rate limit.
        time.sleep(1.0 / 14)


if __name__ == "__main__":
    send_emails()  # pylint: disable=no-value-for-parameter
