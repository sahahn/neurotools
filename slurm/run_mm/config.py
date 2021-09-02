# Don't start a new job if only this
# amount of time left
stop_early = 1000


# Function used in funcs + submit for estimating
def get_job_limit(short=False):

    # Full job time limit
    limit = 108000
    if short:
        limit = 10800

    return limit - stop_early

