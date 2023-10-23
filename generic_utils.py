# For telegram messages
from requests import get

# For progress bar
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display


def send_telegram_message(message):
    """
    Send a Telegram message as a bot to a specificed chat group. In order to send such messages:
    1) A Telegram bot must be initilalized, and a chat must be started with it.
    2) Retrieve chat_id (integer) and bot token (as string) and place them in the 2 homonym variables below.
    Currently the 2 variables are empty because the credentials must be private.

    message (str): content of the message to be sent.
    """
    try:
        chat_id = None
        token = None
        if chat_id == None or token == None:
            print(f"TELEGRAM MESSAGE FAILED: missing chat_id or token. Message content: \n    {message}")
            return
        get(
            "".join(
                [
                    "https://api.telegram.org/bot",
                    str(token),
                    "/sendMessage?chat_id=",
                    str(chat_id),
                    "&text=",
                    str(message),
                ]
            )
        )
    except:
        print(
            "ERROR: can't send telegram message. Check Internet connection. Message print on stdout:"
        )
        print(message)


def log_progress(sequence, every=None, size=None, name="Items"):

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)  # every 0.5%
    else:
        assert every is not None, "sequence is iterator, set every"

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = "info"
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = "{name}: {index} / ?".format(name=name, index=index)
                else:
                    progress.value = (
                        index - 1
                    )  # LAST MODIFICATION -----------------------
                    label.value = "{name}: {index} / {size}".format(
                        name=name, index=index - 1, size=size
                    )
            yield record
    except:
        progress.bar_style = "danger"
        raise
    else:
        progress.bar_style = "success"
        progress.value = index
        label.value = "{name}: {index}".format(name=name, index=str(index or "?"))
