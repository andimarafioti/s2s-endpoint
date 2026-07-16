def monotonic_sequence(*values):
    value_iter = iter(values)

    def fake_monotonic():
        return next(value_iter)

    return fake_monotonic
