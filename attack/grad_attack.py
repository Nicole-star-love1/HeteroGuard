# -*- coding: utf-8 -*-
"""
Gradient attack is kept as an optional alias to CBA-style hybrid attack.

It is not included in the six main attacks by default because true gradient
selection requires a surrogate training loop and is slower. This lightweight
version reuses CBA behavior for compatibility with old "grad/gcba" configs.
"""

from .community_attack import HetCBA


class HetGradBA(HetCBA):
    attack_name = "grad"
