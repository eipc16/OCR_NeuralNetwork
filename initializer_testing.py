from initializers.he_initializer import HeInitializer
from initializers.normal_initializer import NormalInitializer
from initializers.xavier_initializer import XavierInitializer
from tests.initializer_tests import perform_initializer_test

initializers = [
    XavierInitializer(gain=6),
    HeInitializer(),
    NormalInitializer(loc=0, scale=1, a=10)
]

perform_initializer_test(initializers)
