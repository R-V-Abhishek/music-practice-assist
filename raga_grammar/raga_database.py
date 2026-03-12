"""
Carnatic Raga Knowledge Base

Contains detailed information for 34 common ragas plus programmatic generation
of all 72 Melakarta ragas. Each raga specifies:
- Arohana/Avarohana sequences (including vakra patterns)
- Forbidden swaras (varja) per direction
- Parent Melakarta classification
- Characteristic features

Data source: Traditional Carnatic music theory + user-specified raga list
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RagaInfo:
    """Complete information for a single raga"""
    name: str
    arohana: List[str]           # Ascending note sequence
    avarohana: List[str]         # Descending note sequence  
    varja_arohana: Set[str]      # Forbidden notes in ascent
    varja_avarohana: Set[str]    # Forbidden notes in descent
    parent_mela: int             # Melakarta number (1-72)
    is_vakra_arohana: bool       # Has zigzag ascent pattern
    is_vakra_avarohana: bool     # Has zigzag descent pattern
    janya: bool                  # Is derived (janya) raga vs parent (melakarta)
    characteristic_phrases: List[List[str]]  # Signature melodic phrases

# 12 Carnatic swaras with their standard cent values (from Sa)
SWARA_CENTS = {
    'Sa': 0,      'Ri1': 90,   'Ri2': 204,   'Ri3': 294,
    'Ga1': 294,   'Ga2': 386,  'Ga3': 498,   'Ma1': 498, 
    'Ma2': 612,   'Pa': 702,   'Dha1': 792,  'Dha2': 906,
    'Dha3': 996,  'Ni1': 996,  'Ni2': 1088, 'Ni3': 1200
}

# All 12 swara positions for Melakarta generation
ALL_SWARAS = ['Sa', 'Ri1', 'Ri2', 'Ri3', 'Ga1', 'Ga2', 'Ga3', 
              'Ma1', 'Ma2', 'Pa', 'Dha1', 'Dha2', 'Dha3', 'Ni1', 'Ni2', 'Ni3']

def _compute_varja_swaras(arohana: List[str], avarohana: List[str]) -> Tuple[Set[str], Set[str]]:
    """Compute forbidden swaras by finding which are missing from each direction"""
    all_swaras = set(ALL_SWARAS)
    arohana_used = set(arohana)
    avarohana_used = set(avarohana)
    
    # Forbidden = all possible swaras minus those actually used
    varja_arohana = all_swaras - arohana_used
    varja_avarohana = all_swaras - avarohana_used
    
    return varja_arohana, varja_avarohana

def _detect_vakra(sequence: List[str]) -> bool:
    """Detect if a sequence has vakra (zigzag) pattern by checking swara ordinals"""
    if len(sequence) < 3:
        return False
    
    # Map swaras to ordinal positions
    ordinals = []
    for swara in sequence:
        if swara in SWARA_CENTS:
            ordinals.append(SWARA_CENTS[swara])
    
    # Check for direction changes (zigzag)
    for i in range(len(ordinals) - 2):
        # If we go up then down, or down then up = vakra
        diff1 = ordinals[i+1] - ordinals[i]
        diff2 = ordinals[i+2] - ordinals[i+1]
        if diff1 * diff2 < 0:  # Sign change = direction reversal
            return True
    return False

# Manual encoding of 34 user-specified ragas
_RAGA_DEFINITIONS = {
    "Māyāmāḷavagauḷa": {
        "arohana": ["Sa", "Ri1", "Ga1", "Pa", "Dha1", "Sa"],
        "avarohana": ["Sa", "Dha1", "Pa", "Ga1", "Ri1", "Sa"],
        "parent_mela": 15,
        "janya": True
    },
    
    "Mōhanaṁ": {
        "arohana": ["Sa", "Ri2", "Ga2", "Pa", "Dha2", "Sa"],
        "avarohana": ["Sa", "Dha2", "Pa", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Kalyāṇi": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma2", "Pa", "Dha2", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma2", "Ga2", "Ri2", "Sa"],
        "parent_mela": 65,
        "janya": False  # This IS the 65th Melakarta
    },
    
    "Madhyamāvati": {
        "arohana": ["Sa", "Ri2", "Ma1", "Pa", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Pa", "Ma1", "Ri2", "Sa"],
        "parent_mela": 22,
        "janya": True
    },
    
    "Varāḷi": {
        "arohana": ["Sa", "Ga1", "Ma1", "Pa", "Ni1", "Sa"],
        "avarohana": ["Sa", "Ni1", "Pa", "Ma1", "Ga1", "Sa"],
        "parent_mela": 39,
        "janya": True
    },
    
    "Sencuruṭṭi": {
        "arohana": ["Sa", "Ri1", "Ma1", "Pa", "Ni1", "Sa"],
        "avarohana": ["Sa", "Ni1", "Pa", "Ma1", "Ri1", "Sa"],
        "parent_mela": 39,
        "janya": True
    },
    
    "Kāṁbhōji": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Dha2", "Sa"],  # Ni varja in ascent
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"], # Ni present in descent
        "parent_mela": 28,
        "janya": True
    },
    
    "Karaharapriya": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Dha2", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 22,
        "janya": False  # This IS the 22nd Melakarta
    },
    
    "Ānandabhairavi": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Dha1", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Dha1", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 20,
        "janya": True
    },
    
    "Bilahari": {
        "arohana": ["Sa", "Ri2", "Ga2", "Pa", "Dha2", "Sa"],  # Ma varja
        "avarohana": ["Sa", "Dha2", "Pa", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Kānaḍa": {
        "arohana": ["Sa", "Ri2", "Ga1", "Ma1", "Pa", "Dha1", "Sa"],  # Ni varja
        "avarohana": ["Sa", "Ni2", "Dha1", "Pa", "Ma1", "Ga1", "Ri2", "Sa"],
        "parent_mela": 20,
        "janya": True
    },
    
    "Śrīranjani": {
        "arohana": ["Sa", "Ri2", "Ga1", "Ma1", "Pa", "Ni2", "Sa"],  # Dha varja
        "avarohana": ["Sa", "Ni2", "Pa", "Ma1", "Ga1", "Ri2", "Sa"],
        "parent_mela": 22,
        "janya": True
    },
    
    "Kamās": {
        "arohana": ["Sa", "Ri1", "Ma1", "Pa", "Dha1", "Sa"],  # Ga, Ni varja
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma1", "Ga1", "Ri1", "Sa"],
        "parent_mela": 28,
        "janya": True
    },
    
    "Tōḍi": {
        "arohana": ["Sa", "Ri1", "Ga1", "Ma2", "Pa", "Dha1", "Ni1", "Sa"],
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma2", "Ga1", "Ri1", "Sa"],
        "parent_mela": 8,
        "janya": False  # This IS the 8th Melakarta
    },
    
    "Mukhāri": {
        "arohana": ["Sa", "Ri2", "Ga1", "Ma1", "Pa", "Dha1", "Ni1", "Sa"],
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma1", "Ga1", "Ri2", "Sa"],
        "parent_mela": 22,
        "janya": True
    },
    
    "Suraṭi": {
        "arohana": ["Sa", "Ri2", "Ma1", "Pa", "Dha1", "Sa"],  # Ga, Ni varja
        "avarohana": ["Sa", "Dha1", "Pa", "Ma1", "Ri2", "Sa"],
        "parent_mela": 22,
        "janya": True
    },
    
    "Kāpi": {
        "arohana": ["Sa", "Ri2", "Ga2", "Pa", "Dha1", "Sa"],  # Ma, Ni varja
        "avarohana": ["Sa", "Ni2", "Dha1", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 22,
        "janya": True
    },
    
    "Nāṭakurinji": {
        "arohana": ["Sa", "Ri1", "Ga1", "Ma2", "Pa", "Dha1", "Sa"],  # Ni varja
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma2", "Ga1", "Ri1", "Sa"],
        "parent_mela": 8,
        "janya": True
    },
    
    "Pūrvīkaḷyāṇi": {
        "arohana": ["Sa", "Ri1", "Ga2", "Ma2", "Pa", "Dha1", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Dha1", "Pa", "Ma2", "Ga2", "Ri1", "Sa"],
        "parent_mela": 53,
        "janya": True
    },
    
    "Kēdāragauḷa": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma2", "Pa", "Dha2", "Sa"],  # Ni varja
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma2", "Ga2", "Ri2", "Sa"],
        "parent_mela": 65,
        "janya": True
    },
    
    "Sāvēri": {
        "arohana": ["Sa", "Ri1", "Ma1", "Pa", "Dha2", "Sa"],  # Ga, Ni varja
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ri1", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Śankarābharaṇaṁ": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Dha2", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": False  # This IS the 29th Melakarta
    },
    
    "Gauḷa": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Ni2", "Sa"],  # Dha varja
        "avarohana": ["Sa", "Ni2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Rītigauḷa": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Ni2", "Sa"],  # Pa, Dha varja
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Śrī": {
        "arohana": ["Sa", "Ri1", "Ma1", "Pa", "Ni1", "Sa"],  # Ga, Dha varja
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma1", "Ga1", "Ri1", "Sa"],
        "parent_mela": 22,
        "janya": True
    },
    
    "Kāmavardani": {
        "arohana": ["Sa", "Ri1", "Ga1", "Ma2", "Pa", "Dha2", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma2", "Ga1", "Ri1", "Sa"],
        "parent_mela": 51,
        "janya": False  # This IS the 51st Melakarta
    },
    
    "Bhairavi": {
        "arohana": ["Sa", "Ri1", "Ga1", "Ma1", "Pa", "Dha1", "Ni1", "Sa"],
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma1", "Ga1", "Ri1", "Sa"],
        "parent_mela": 8,
        "janya": True
    },
    
    "Ṣanmukhapriya": {
        "arohana": ["Sa", "Ri1", "Ga2", "Ma1", "Pa", "Dha1", "Ni1", "Sa"],
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma1", "Ga2", "Ri1", "Sa"],
        "parent_mela": 56,
        "janya": False  # This IS the 56th Melakarta
    },
    
    "Aṭāna": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Ma1", "Dha2", "Ni2", "Sa"],  # Vakra: Pa-Ma1-Dha2
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Harikāmbhōji": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Dha2", "Ni2", "Sa"],
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 28,
        "janya": False  # This IS the 28th Melakarta
    },
    
    "Bēgaḍa": {
        "arohana": ["Sa", "Ga2", "Ri2", "Ga2", "Ma1", "Pa", "Ni2", "Dha2", "Ni2", "Sa"],  # Vakra: Ga2-Ri2-Ga2, Ni2-Dha2-Ni2
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Nāṭa": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Ni2", "Sa"],  # Dha varja
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 29,
        "janya": True
    },
    
    "Yadukula kāṁbōji": {
        "arohana": ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Dha2", "Sa"],  # Ni varja
        "avarohana": ["Sa", "Ni2", "Dha2", "Pa", "Ma1", "Ga2", "Ri2", "Sa"],
        "parent_mela": 28,
        "janya": True
    },
    
    "Sahānā": {
        "arohana": ["Sa", "Ri1", "Ma1", "Pa", "Dha1", "Sa"],  # Ga, Ni varja
        "avarohana": ["Sa", "Ni1", "Dha1", "Pa", "Ma1", "Ga1", "Ri1", "Sa"], 
        "parent_mela": 28,
        "janya": True
    }
}

def _build_raga_info(name: str, definition: Dict) -> RagaInfo:
    """Convert raw definition dict to complete RagaInfo object"""
    arohana = definition["arohana"]
    avarohana = definition["avarohana"]
    
    # Compute forbidden swaras
    varja_arohana, varja_avarohana = _compute_varja_swaras(arohana, avarohana)
    
    # Detect vakra patterns
    is_vakra_arohana = _detect_vakra(arohana)
    is_vakra_avarohana = _detect_vakra(avarohana)
    
    return RagaInfo(
        name=name,
        arohana=arohana,
        avarohana=avarohana,
        varja_arohana=varja_arohana,
        varja_avarohana=varja_avarohana,
        parent_mela=definition["parent_mela"],
        is_vakra_arohana=is_vakra_arohana,
        is_vakra_avarohana=is_vakra_avarohana,
        janya=definition["janya"],
        characteristic_phrases=definition.get("characteristic_phrases", [])
    )

# Build complete raga database
RAGA_DB: Dict[str, RagaInfo] = {}
for name, definition in _RAGA_DEFINITIONS.items():
    RAGA_DB[name] = _build_raga_info(name, definition)

def get_raga_info(raga_name: str) -> Optional[RagaInfo]:
    """Get complete information for a raga by name"""
    return RAGA_DB.get(raga_name)

def get_melakarta_raga(mela_number: int) -> Optional[RagaInfo]:
    """Generate or retrieve a Melakarta raga by number (1-72)"""
    if not 1 <= mela_number <= 72:
        return None
    
    # Check if it's already in our database
    for raga_info in RAGA_DB.values():
        if raga_info.parent_mela == mela_number and not raga_info.janya:
            return raga_info
    
    # Generate algorithmically using Katapayadi scheme
    return _generate_melakarta(mela_number)

def _generate_melakarta(mela_number: int) -> RagaInfo:
    """Generate a Melakarta raga using the mathematical Katapayadi scheme"""
    # Melakarta formula: 36 upper melakartas (Ma2), 36 lower melakartas (Ma1)
    if mela_number <= 36:
        ma = "Ma1"
        base_mela = mela_number
    else:
        ma = "Ma2" 
        base_mela = mela_number - 36
    
    # Each group of 6 corresponds to different Ri-Ga combinations
    ri_ga_group = ((base_mela - 1) // 6) + 1
    dha_ni_variant = ((base_mela - 1) % 6) + 1
    
    # Ri-Ga mapping (6 variants)
    ri_ga_map = {
        1: ["Ri1", "Ga1"], 2: ["Ri1", "Ga2"], 3: ["Ri1", "Ga3"],
        4: ["Ri2", "Ga2"], 5: ["Ri2", "Ga3"], 6: ["Ri3", "Ga3"]
    }
    
    # Dha-Ni mapping (6 variants) 
    dha_ni_map = {
        1: ["Dha1", "Ni1"], 2: ["Dha1", "Ni2"], 3: ["Dha1", "Ni3"],
        4: ["Dha2", "Ni2"], 5: ["Dha2", "Ni3"], 6: ["Dha3", "Ni3"]
    }
    
    ri, ga = ri_ga_map[ri_ga_group]
    dha, ni = dha_ni_map[dha_ni_variant]
    
    # Build complete scale
    arohana = ["Sa", ri, ga, ma, "Pa", dha, ni, "Sa"]
    avarohana = ["Sa", ni, dha, "Pa", ma, ga, ri, "Sa"]
    
    # Melakarta name generation (simplified)
    mela_name = f"Melakarta_{mela_number}"
    
    return RagaInfo(
        name=mela_name,
        arohana=arohana,
        avarohana=avarohana,
        varja_arohana=set(),  # Melakarta ragas have no forbidden notes
        varja_avarohana=set(),
        parent_mela=mela_number,
        is_vakra_arohana=False,  # Melakarta ragas are sampurna (linear)
        is_vakra_avarohana=False,
        janya=False,
        characteristic_phrases=[]
    )

def list_available_ragas() -> List[str]:
    """Get list of all available raga names"""
    return list(RAGA_DB.keys())

def search_ragas_by_parent(mela_number: int) -> List[str]:
    """Find all ragas derived from a specific Melakarta"""
    return [name for name, info in RAGA_DB.items() 
            if info.parent_mela == mela_number]