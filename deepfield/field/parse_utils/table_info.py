"""Table meta-information required for parsing"""

TABLE_INFO = {
    'PVTO': dict(attrs=['RS', 'PRESSURE', 'FVF', 'VISC'], domain=[0, 1],
                 defaults=None),

    'PVTG': dict(attrs=['PRESSURE', 'RV', 'FVF', 'VISC'], domain=[0, 1],
                 defaults=None),

    'PVDG': dict(attrs=['PRESSURE', 'FVF', 'VISC'], domain=[0],
                 defaults=None),

    'PVTW': dict(attrs=['PRESSURE', 'FVF', 'COMPR', 'VISC', 'VISCOSIBILITY'], domain=[0],
                 defaults=[None, None, None, None, 0]),

    'SWOF': dict(attrs=['SW', 'KRWO', 'KROW', 'POW'], domain=[0],
                 defaults=[None, None, None, 0]),

    'SGOF': dict(attrs=['SG', 'KRGO', 'KROG', 'POG'], domain=[0],
                 defaults=[None, None, None, 0]),

    'RSVD': dict(attrs=['DEPTH', 'RS'], domain=[0],
                 defaults=None),

    'ROCK': dict(attrs=['PRESSURE', 'COMPR'], domain=[0],
                 defaults=None),

    'DENSITY': dict(attrs=['DENSO', 'DENSW', 'DENSG'], domain=None,
                    defaults=[600, 999.014, 1])

}
