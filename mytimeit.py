import __main__
import timeit

def mytimeit(stmt, n=5):
    min_elapsed = min([
        timeit.timeit(stmt=stmt, number=1, globals=__main__.__dict__)
        for i in range(n)])
    print(f'{stmt}: quickest out of {n} took {min_elapsed} s')

# vim: ts=4 sw=4 et ai sta
