from .chain import Chain
from .chain_4 import Chain4
from .chain_4_clean import Chain4_Clean
from .chain_5 import Chain5
from .chain_5_clean import Chain5_Clean
from .chain_10 import Chain10
from .triangle import Triangle
from .collider import Collider
from .fork import Fork
from .diamond import Diamond
from .simpson import Simpson
from .large_backdoor import LargeBackdoor
from .german_credit import GermanCredit
from .george_test import George_Test
from .george_int import George_Int
from .modified_diamond import ModifiedDiamond
from .modified_fork_chain import ModifiedForkChain
from .weird_chain import Weird_chain
from .weird_chain_confounder_2_3 import Weird_chain_z23
from .weird_chain_confounder_3_2 import Weird_chain_z32
from .chain_confounded import ChainConfounded
from .chain_2 import Chain2

sem_dict = {}

sem_dict["chain"] = Chain
sem_dict["chain-confounded"] = ChainConfounded
sem_dict["chain-2"] = Chain2
sem_dict["chain-4"] = Chain4
sem_dict["chain-5"] = Chain5
sem_dict["chain-10"] = Chain10
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
sem_dict["chain-4-clean"] = Chain4_Clean
sem_dict["modified-diamond"] = ModifiedDiamond
sem_dict["modified-fork-chain"] = ModifiedForkChain
sem_dict["weird_chain"] = Weird_chain
sem_dict["weird_chain_z23"] = Weird_chain_z23
sem_dict["weird_chain_z32"] = Weird_chain_z32