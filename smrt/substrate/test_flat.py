from smrt.inputs.make_soil import make_soil


def test_make_flat():
    make_soil("flat", "dobson85_peplinski95", 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100)
