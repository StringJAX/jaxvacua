# Copyright 2022 Andreas Schachner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
**Description:**
JAXVacua: A library for analysing string compactifications and constructing string vacua.

Sub-modules:
	periods: Contains `periods` class implementing standard formulas in terms of the periods.
	css: Contains `css` class containing functions to handle
		the Kähler geometry of the complex structure sector.
	flux_sector: Contains `flux_sector` class for computations of the flux scalar potential induced by 
		3-form flux backgrounds.
	util: Contains utility functions.
	axion_eft: Computes masses, decay constants and axion-photon couplings in CY compactifications.

"""


from .util import *
#from .utils_jaxvacua import *
from .cytools_interface import *
from .lcs_init import *
from .periods_LCS import *
from .coniLCS_init import *
from .periods_coniLCS import *
from .periods_coniLCSbulk import *
from .periods import *
from .css_LCS import *
from .css_coniLCS import *
from .css_coniLCSbulk import *
from .css import *
from .flux_sector import *
from .flux_eft import *
from .sampling import *
from .basics import *
from .matrix_cut import *



__version__ = '0.0.1'
