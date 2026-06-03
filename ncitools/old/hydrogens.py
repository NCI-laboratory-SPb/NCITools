from ase import Atoms
from ase.data import atomic_numbers, covalent_radii, atomic_names
import numpy as np
from scipy.spatial import distance_matrix
from collections import defaultdict
import math


class HydrogenAdder:
    def __init__(self, bond_tolerance=0.4, angle_tolerance=15.0):
        """
        Инициализация класса для добавления водородов

        Parameters:
        bond_tolerance: допуск для определения связей (в ангстремах)
        angle_tolerance: допуск для определения углов (в градусах)
        """
        self.bond_tolerance = bond_tolerance
        self.angle_tolerance = angle_tolerance

        # Стандартные валентности для элементов
        self.standard_valence = {
            'H': 1,
            'C': 4, 'Si': 4, 'Ge': 4,
            'N': 3, 'P': 3, 'As': 3,
            'O': 2, 'S': 2, 'Se': 2, 'Te': 2,
            'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
            'B': 3, 'Al': 3,
            'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
            'Zn': 2, 'Cd': 2, 'Hg': 2
        }

        # Максимальные валентности (для переходных металлов)
        self.max_valence = {
            'Fe': 6, 'Co': 6, 'Ni': 6, 'Cu': 6, 'Zn': 6,
            'Cr': 6, 'Mn': 6, 'V': 6, 'Ti': 6, 'Sc': 6,
            'Pt': 6, 'Pd': 6, 'Au': 6, 'Ag': 6
        }

    def calculate_bonds(self, atoms, atom_index):
        """
        Вычисляет количество связей и соседей для атома
        на основе геометрических критериев
        """
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        current_pos = positions[atom_index]
        current_symbol = symbols[atom_index]

        # Ковалентный радиус текущего атома
        try:
            current_radius = covalent_radii[atomic_numbers[current_symbol]]
        except (KeyError, IndexError):
            current_radius = 1.5  # значение по умолчанию

        bonds = 0
        bonded_atoms = []
        bond_lengths = []

        for i, (pos, symbol) in enumerate(zip(positions, symbols)):
            if i == atom_index:
                continue

            try:
                other_radius = covalent_radii[atomic_numbers[symbol]]
            except (KeyError, IndexError):
                other_radius = 1.5  # значение по умолчанию

            bond_length = np.linalg.norm(current_pos - pos)
            expected_bond_length = current_radius + other_radius

            # Проверяем, находится ли расстояние в пределах допуска
            if bond_length <= expected_bond_length + self.bond_tolerance:
                bonds += 1
                bonded_atoms.append(i)
                bond_lengths.append(bond_length)

        return bonds, bonded_atoms, bond_lengths

    def calculate_bond_angles(self, atoms, atom_index, bonded_atoms):
        """
        Вычисляет валентные углы для атома
        """
        if len(bonded_atoms) < 2:
            return []

        positions = atoms.get_positions()
        current_pos = positions[atom_index]

        angles = []

        # Вычисляем все попарные углы
        for i in range(len(bonded_atoms)):
            for j in range(i + 1, len(bonded_atoms)):
                vec1 = positions[bonded_atoms[i]] - current_pos
                vec2 = positions[bonded_atoms[j]] - current_pos

                # Нормализуем векторы
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)

                # Вычисляем угол в градусах
                dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))
                angles.append(angle)

        return angles

    def determine_hybridization(self, bond_angles):
        """
        Определяет тип гибридизации на основе валентных углов
        """
        if not bond_angles:
            return 'unknown'

        avg_angle = np.mean(bond_angles)

        # Критерии для определения гибридизации
        if len(bond_angles) == 1:
            return 'sp'  # линейная для двух атомов

        # Для 2 связей
        elif len(bond_angles) == 1:  # один угол для 2 связей
            if 160 <= avg_angle <= 180:
                return 'sp'
            elif 100 <= avg_angle <= 120:
                return 'sp2'
            elif 80 <= avg_angle <= 110:
                return 'sp3'

        # Для 3 и более связей - анализируем все углы
        else:
            angles_sorted = sorted(bond_angles)

            # Проверяем на sp (линейная)
            if all(160 <= angle <= 180 for angle in angles_sorted):
                return 'sp'

            # Проверяем на sp2 (тригональная)
            elif all(115 <= angle <= 125 for angle in angles_sorted):
                return 'sp2'

            # Проверяем на sp3 (тетраэдрическая)
            elif all(100 <= angle <= 112 for angle in angles_sorted):
                return 'sp3'

            # Для сложных случаев используем средний угол
            else:
                if avg_angle >= 150:
                    return 'sp'
                elif avg_angle >= 110:
                    return 'sp2'
                else:
                    return 'sp3'

    def get_expected_valence(self, atom_symbol, bonds, hybridization='unknown'):
        """
        Определяет ожидаемую валентность для атома
        с учетом его типа и гибридизации
        """
        # Базовые валентности
        if atom_symbol in self.standard_valence:
            base_valence = self.standard_valence[atom_symbol]
        else:
            # Для неизвестных элементов используем эвристику
            base_valence = min(bonds + 2, 8)  # правило октета

        # Корректировки для конкретных элементов и гибридизаций
        if atom_symbol == 'C':
            if hybridization == 'sp3':
                return 4
            elif hybridization == 'sp2':
                return 3
            elif hybridization == 'sp':
                return 2

        elif atom_symbol == 'N':
            if hybridization == 'sp3':
                return 3
            elif hybridization == 'sp2':
                return 2  # или 3 для нитрилов
            elif hybridization == 'sp':
                return 1  # или 3 для нитрилов

        elif atom_symbol == 'O':
            return 2

        elif atom_symbol == 'S':
            if bonds >= 4:  # сульфоны, сульфоксиды
                return 6
            else:
                return 2

        # Для переходных металлов
        if atom_symbol in self.max_valence:
            return min(self.max_valence[atom_symbol], base_valence)

        return base_valence

    def generate_hydrogen_positions(self, atoms, atom_index, num_hydrogens, bonded_atoms):
        """
        Генерирует позиции для атомов водорода с учетом гибридизации
        """
        if num_hydrogens <= 0:
            return []

        positions = atoms.get_positions()
        current_pos = positions[atom_index]
        symbol = atoms.get_chemical_symbols()[atom_index]

        # Определяем гибридизацию для лучшего размещения
        bond_angles = self.calculate_bond_angles(atoms, atom_index, bonded_atoms)
        hybridization = self.determine_hybridization(bond_angles)

        # Стандартные длины связей
        bond_lengths = {
            'C-H': 1.09, 'N-H': 1.01, 'O-H': 0.96, 'S-H': 1.34,
            'P-H': 1.42, 'Si-H': 1.48, 'default': 1.0
        }

        bond_key = f"{symbol}-H"
        if bond_key in bond_lengths:
            bond_length = bond_lengths[bond_key]
        else:
            bond_length = bond_lengths['default']

        hydrogen_positions = []

        if len(bonded_atoms) == 0:
            # Изолированный атом
            if num_hydrogens == 1:
                pos = current_pos + [bond_length, 0, 0]
                hydrogen_positions.append(pos)
            elif num_hydrogens == 2:
                if hybridization == 'sp':
                    pos1 = current_pos + [bond_length, 0, 0]
                    pos2 = current_pos + [-bond_length, 0, 0]
                    hydrogen_positions.extend([pos1, pos2])
                else:
                    angle = 104.5  # как у воды
                    pos1 = current_pos + [bond_length, 0, 0]
                    pos2 = current_pos + [bond_length * np.cos(np.radians(angle)),
                                          bond_length * np.sin(np.radians(angle)), 0]
                    hydrogen_positions.extend([pos1, pos2])
            else:
                # Тетраэдрическое расположение
                for i in range(num_hydrogens):
                    theta = np.arccos(-1 + 2 * (i + 0.5) / num_hydrogens)
                    phi = np.sqrt(num_hydrogens * np.pi) * theta
                    x = bond_length * np.sin(theta) * np.cos(phi)
                    y = bond_length * np.sin(theta) * np.sin(phi)
                    z = bond_length * np.cos(theta)
                    hydrogen_positions.append(current_pos + [x, y, z])

        elif len(bonded_atoms) == 1:
            # Линейная или изогнутая геометрия
            bond_vector = positions[bonded_atoms[0]] - current_pos
            bond_vector = bond_vector / np.linalg.norm(bond_vector)

            if hybridization == 'sp' or num_hydrogens == 1:
                # Линейное расположение
                pos = current_pos - bond_length * bond_vector
                hydrogen_positions.append(pos)
            else:
                # Тригональное или тетраэдрическое
                # Находим перпендикулярный вектор
                if abs(bond_vector[2]) < 0.9:
                    perp = np.cross(bond_vector, [0, 0, 1])
                else:
                    perp = np.cross(bond_vector, [1, 0, 0])
                perp = perp / np.linalg.norm(perp)

                if num_hydrogens == 2:
                    # Тригональная плоская
                    angles = [0, 120, 240]
                    for i in range(num_hydrogens):
                        angle_rad = np.radians(angles[i])
                        direction = (np.cos(angle_rad) * (-bond_vector) +
                                     np.sin(angle_rad) * perp)
                        direction = direction / np.linalg.norm(direction)
                        pos = current_pos + bond_length * direction
                        hydrogen_positions.append(pos)
                else:
                    # Тетраэдрическое
                    for i in range(num_hydrogens):
                        angle = 109.5 * i
                        direction = (np.cos(np.radians(angle)) * (-bond_vector) +
                                     np.sin(np.radians(angle)) * perp)
                        direction = direction / np.linalg.norm(direction)
                        pos = current_pos + bond_length * direction
                        hydrogen_positions.append(pos)

        else:
            # Множественные связи - используем направление, противоположное
            # сумме векторов существующих связей
            sum_vector = np.zeros(3)
            for bonded_idx in bonded_atoms:
                bond_vec = positions[bonded_idx] - current_pos
                bond_vec = bond_vec / np.linalg.norm(bond_vec)
                sum_vector += bond_vec

            if np.linalg.norm(sum_vector) > 0.1:
                direction = -sum_vector / np.linalg.norm(sum_vector)
            else:
                direction = np.array([1, 0, 0])  # fallback

            for i in range(num_hydrogens):
                # Немного разбрасываем направления для нескольких H
                if i > 0:
                    rotation_angle = np.radians(15 * i)
                    rot_matrix = self.rotation_matrix(rotation_angle, [0, 0, 1])
                    direction = np.dot(rot_matrix, direction)

                pos = current_pos + bond_length * direction
                hydrogen_positions.append(pos)

        return hydrogen_positions

    def rotation_matrix(self, angle, axis):
        """Создает матрицу поворота вокруг оси"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        one_minus_cos = 1 - cos_a

        return np.array([
            [cos_a + axis[0] * axis[0] * one_minus_cos,
             axis[0] * axis[1] * one_minus_cos - axis[2] * sin_a,
             axis[0] * axis[2] * one_minus_cos + axis[1] * sin_a],
            [axis[1] * axis[0] * one_minus_cos + axis[2] * sin_a,
             cos_a + axis[1] * axis[1] * one_minus_cos,
             axis[1] * axis[2] * one_minus_cos - axis[0] * sin_a],
            [axis[2] * axis[0] * one_minus_cos - axis[1] * sin_a,
             axis[2] * axis[1] * one_minus_cos + axis[0] * sin_a,
             cos_a + axis[2] * axis[2] * one_minus_cos]
        ])

    def add_hydrogens(self, atoms):
        """
        Основная функция для добавления водородов ко всем атомам
        """
        new_atoms = atoms.copy()
        atoms_to_add = []

        print("Анализ молекулы:")
        print("=" * 50)

        for i, atom in enumerate(atoms):
            symbol = atom.symbol

            # Вычисляем текущие связи
            current_bonds, bonded_atoms, bond_lengths = self.calculate_bonds(atoms, i)

            # Вычисляем валентные углы и гибридизацию
            bond_angles = self.calculate_bond_angles(atoms, i, bonded_atoms)
            hybridization = self.determine_hybridization(bond_angles)

            # Определяем ожидаемую валентность
            expected_valence = self.get_expected_valence(symbol, current_bonds, hybridization)

            # Вычисляем необходимое количество водородов
            missing_hydrogens = expected_valence - current_bonds

            # Не добавляем водороды к уже насыщенным атомам или металлам
            if missing_hydrogens > 0 and symbol not in ['Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Cr', 'Mn', 'V', 'Ti']:
                print(f"Атом {symbol}{i + 1}:")
                print(f"  Текущие связи: {current_bonds}")
                print(f"  Связанные атомы: {bonded_atoms}")
                print(f"  Валентные углы: {['%.1f°' % ang for ang in bond_angles]}")
                print(f"  Гибридизация: {hybridization}")
                print(f"  Ожидаемая валентность: {expected_valence}")
                print(f"  Требуется водородов: {missing_hydrogens}")

                hydrogen_positions = self.generate_hydrogen_positions(
                    atoms, i, missing_hydrogens, bonded_atoms
                )

                for pos in hydrogen_positions:
                    atoms_to_add.append(('H', pos))

            else:
                print(f"Атом {symbol}{i + 1}: связи={current_bonds}, гибридизация={hybridization} - насыщен")

        # Добавляем все водороды
        for symbol, position in atoms_to_add:
            new_atoms.append(atomic_numbers[symbol])
            new_atoms.positions[-1] = position

        print(f"\nДобавлено {len(atoms_to_add)} атомов водорода")
        return new_atoms


# Примеры использования
def test_molecules():
    """Тестовые молекулы для демонстрации"""

    # Метан (только углерод)
    methane = Atoms('C', positions=[[0, 0, 0]])

    # Вода (только кислород)
    water = Atoms('O', positions=[[0, 0, 0]])

    # Аммиак (только азот)
    ammonia = Atoms('N', positions=[[0, 0, 0]])

    # Этилен (C2H4 без водородов)
    ethylene = Atoms('CC', positions=[[0, 0, 0], [1.33, 0, 0]])

    # Ацетилен (C2H2 без водородов)
    acetylene = Atoms('CC', positions=[[0, 0, 0], [1.20, 0, 0]])

    # Серная кислота (S с 4 кислородами)
    sulfuric_acid = Atoms('SOOOO', positions=[
        [0, 0, 0],
        [1.43, 0, 0],
        [-1.43, 0, 0],
        [0, 1.43, 0],
        [0, 0, 1.43]
    ])

    return methane, water, ammonia, ethylene, acetylene, sulfuric_acid


# Демонстрация
if __name__ == "__main__":
    hydrogen_adder = HydrogenAdder()

    molecules = test_molecules()
    names = ["Метан", "Вода", "Аммиак", "Этилен", "Ацетилен", "Серная кислота"]

    for name, mol in zip(names, molecules):
        print(f"\n{'=' * 60}")
        print(f"Молекула: {name}")
        print(f"{'=' * 60}")

        result = hydrogen_adder.add_hydrogens(mol)
        print(f"Результат: {mol.get_chemical_formula()} → {result.get_chemical_formula()}")
        print(f"Всего атомов: {len(result)}")