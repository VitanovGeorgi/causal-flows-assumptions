from .chain import Chain
from .chain_4 import Chain4
from .chain_5 import Chain5
from .triangle import Triangle
from .collider import Collider
from .fork import Fork
from .diamond import Diamond
from .simpson import Simpson
from .large_backdoor import LargeBackdoor
from .german_credit import GermanCredit
from .george_test import George_Test
from .george_int import George_Int
from .chain_5_clean import Chain5_Clean

sem_dict = {}

sem_dict["chain"] = Chain
sem_dict["chain-4"] = Chain4
sem_dict["chain-5"] = Chain5
sem_dict["triangle"] = Triangle
sem_dict["collider"] = Collider
sem_dict["fork"] = Fork
sem_dict["diamond"] = Diamond
sem_dict["simpson"] = Simpson
sem_dict["large-backdoor"] = LargeBackdoor
sem_dict["german"] = GermanCredit
sem_dict["george_test"] = George_Test
sem_dict["george_int"] = George_Int
sem_dict["chain-5-clean"] = Chain5_Clean
