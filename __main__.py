import re
from collections import defaultdict
import itertools
import functools
import random
from pathlib import Path
import traceback
import numpy as np
from tkinter import filedialog


class Cell:
    def __init__(self, x: int, y: int, val: int, cands: set[int] = set()) -> None:
        self.x = x
        self.y = y
        self.val = val
        self._cands = cands

    def copy(self):
        return Cell(self.x, self.y, self.val, self._cands.copy())

    @property
    def loc(self):
        return self.x, self.y

    @property
    def z(self):
        return 3 * (self.x // 3) + (self.y // 3)

    @property
    def cands(self):
        if self.val:
            self._cands.clear()
        return self._cands

    def remove_cands(self, *cands: int):
        self._cands -= set(cands)
        return self._cands

    def __hash__(self) -> int:
        return hash(self.val)

    def __iter__(self):
        return iter(self.cands)

    def __len__(self):
        return len(self.cands)

    def eq_by_val(self, other) -> bool:
        return hash(self) == hash(other)  #! eq by val

    def eq_by_loc(self, other) -> bool:
        try:
            return self.loc == other.loc  #! eq by location
        except:
            return self.eq_by_val(other)

    def __eq__(self, *x):
        raise NotImplementedError

    def wow(self) -> str:
        return f"{self.loc}={self.val or self.cands}"

    def print_wow_with_check(self):
        if not self.val:
            print(self.loc, self._cands, sep="=")

    def __repr__(self) -> str:
        return str(self.val or "+")
        # return self.wow()
        # return str(self.val or self.cands)


class FishMultiple:
    def fast_fish_multiple(self, ple, finned=False):
        popped = []
        num = ple**2
        for rows_cols in itertools.product(
            itertools.combinations(range(9), ple), repeat=2
        ):
            pattern: set[Cell] = set()
            acceptance: tuple[dict[int, int]] = tuple(
                defaultdict(int) for _ in range(2)
            )
            subgrids: set[int] = set()
            same_cands: set[int] = self.NUMS.copy()  # all values
            entered = 0
            for cords in itertools.product(*rows_cols):
                if (cell := self.mat[cords]).cands:
                    same_cands &= cell._cands
                    if not same_cands:
                        break
                    entered += 1
                    for acc, cord in zip(acceptance, cords):
                        acc[cord] += 1
                    subgrids.add(cell.z)
                Cell.__eq__ = Cell.eq_by_loc
                pattern.add(cell)
            if entered < 2 * ple:
                continue
            if not same_cands:
                continue
            if len(pattern) != num:
                continue
            for f_i, (f1, f2) in enumerate(
                itertools.permutations((self.row, self.col), 2)
            ):
                formation = tuple(acceptance[f_i].values())
                if len(formation) < ple:
                    continue
                elif any(ff < 2 for ff in formation):
                    continue
                q1, q2 = f1(*acceptance[f_i]), f2(*acceptance[1 - f_i])
                for fish_digit in same_cands:
                    Cell.__eq__ = Cell.eq_by_loc
                    fin_cells_subg: set[Cell] = set()
                    if fins := (set(c for c in q1 if fish_digit in c.cands) - pattern):
                        if finned:
                            fin_subgrids: set[int] = set(fin.z for fin in fins)
                            if len(fin_subgrids) == 1 and (fin_subgrids & subgrids):
                                fin_cells_subg = set(self.subgrid(fin_subgrids.pop()))
                        elif not finned:
                            continue
                    if finned and not fin_cells_subg:
                        continue
                    for fished in set(c for c in q2 if fish_digit in c.cands) - pattern:
                        if not finned or (fished in fin_cells_subg):
                            # print("sashimi") if sashimi else ...
                            fished.remove_cands(fish_digit)
                            popped.append(fished)
                if popped:
                    return popped
        return popped

    def slow_fish_multiple(self, ple, finned=False):
        popped = []
        num = ple**2
        for rows_cols in itertools.product(
            itertools.combinations(range(9), ple), repeat=2
        ):
            pattern: set[Cell] = set(
                self.mat[cords] for cords in itertools.product(*rows_cols)
            )
            subgrids: set[int] = set()
            cand_to_cells: dict[int, set[Cell]] = defaultdict(set)
            cand_to_acceptance = defaultdict(
                lambda: tuple(defaultdict(int) for _ in range(2))
            )
            Cell.__eq__ = Cell.eq_by_loc
            filled_cells: set[Cell] = set()
            for cell in pattern:
                if cell.cands:
                    for cand in cell._cands:
                        cand_to_cells[cand].add(cell)
                    subgrids.add(cell.z)
                else:
                    filled_cells.add(cell)
                Cell.__eq__ = Cell.eq_by_loc
            to_view_fishes: list[tuple[int, int]] = []
            for fish_digit, the_cells in cand_to_cells.items():
                ltc, lfc = len(the_cells), len(filled_cells)
                if ltc + lfc == num:  # cand + filled == 9
                    to_view_fishes.append((fish_digit, 0))
                elif ltc >= 2 * ple:
                    to_view_fishes.append((fish_digit, 0))
                elif ltc < 2 * ple:
                    to_view_fishes.append((fish_digit, 2 * ple - ltc))
                else:
                    continue
                acceptance = cand_to_acceptance[fish_digit]
                for cell in the_cells:
                    for acc, cord in zip(acceptance, cell.loc):
                        acc[cord] += 1
            if not to_view_fishes:
                continue
            for fish_digit, needs_this_many in to_view_fishes:
                acceptance = cand_to_acceptance[fish_digit]
                for f_i, (f1, f2) in enumerate(
                    itertools.permutations((self.row, self.col), 2)
                ):
                    formation = tuple(acceptance[f_i].values())
                    sashimi = False
                    if len(formation) < ple:
                        continue
                    elif any(ff < 2 for ff in formation):
                        if finned and formation.count(1) == 1:
                            other_cells = pattern - cand_to_cells[fish_digit]
                            if len(other_cells) == 1 and other_cells & filled_cells:
                                sashimi = True
                            else:
                                continue
                        else:
                            continue
                    q1, q2 = f1(*acceptance[f_i]), f2(*acceptance[1 - f_i])
                    Cell.__eq__ = Cell.eq_by_loc
                    fin_cells_subg: set[Cell] = set()
                    fins = set(c for c in q1 if fish_digit in c.cands) - pattern
                    if fins:
                        if finned:
                            fin_subgrids: set[int] = set(fin.z for fin in fins)
                            if len(fin_subgrids) == 1 and (fin_subgrids & subgrids):
                                fin_cells_subg = set(self.subgrid(fin_subgrids.pop()))
                        else:
                            continue
                    if finned and not fin_cells_subg:
                        continue
                    fished_cells = set(c for c in q2 if fish_digit in c.cands) - pattern
                    if finned:
                        fished_cells &= fin_cells_subg
                    for fished in fished_cells:
                        fished.remove_cands(fish_digit)
                        popped.append(fished)
                if popped:
                    return popped
        return popped


class Sudoku(FishMultiple):
    NUMS = set(range(1, 10))

    def __init__(self, matrix: np.ndarray, compute=True) -> None:
        self.mat: np.ndarray = matrix
        if compute:
            self.compute_candidates()
            assert self.good

    def copy(self):
        return Sudoku(
            np.array([c.copy() for c in self.mat.flatten()], dtype=Cell).reshape(
                (9, 9)
            ),
            compute=False,
        )

    def __str__(self):
        return str(self.mat)

    @classmethod
    def from_file(cls, path: Path):
        return cls.from_string(path.read_text())

    @classmethod
    def from_string(cls, text: str):
        text = re.sub(r"\+|-|\|", " ", text)
        text = re.sub(r"\s", "", text)
        text = re.sub(r"_|\.", "0", text)
        arr = np.array(
            [Cell(*divmod(ii, 9), int(val)) for ii, val in enumerate(text)],
            dtype=Cell,
        ).reshape((9, 9))
        return cls(arr), text

    def subgrid(self, n: int):
        row, col = map((3).__mul__, divmod(n, 3))
        return self.mat[row : row + 3, col : col + 3].flatten()

    def row(self, *rows: tuple[int]):
        return self.mat[rows, :].flatten()

    def col(self, *cols: tuple[int]):
        return self.mat[:, cols].flatten()

    def region(self, cell: Cell, *, by) -> set[Cell]:
        Cell.__eq__ = by
        res: set[Cell] = (
            set(self.row(cell.x)) | set(self.col(cell.y)) | set(self.subgrid(cell.z))
        )
        return res

    @staticmethod
    def info_deco(func):
        @functools.wraps(func)
        def outer(show):
            def inner(*args, **kwargs):
                res = func(*args, **kwargs)
                if res and show:
                    name = func.__name__.replace(*"_ ")
                    if "backtracking" == name:
                        Sudoku.LAST = []
                        print("\033[91m", end="")  #! set color to red
                    print(
                        f"{name!r:25}",
                        "->",
                        *((getattr(rr, "wow", None) or rr.__str__)() for rr in res),
                        sep=" ",
                    )
                    if "backtracking" == name:
                        print("\033[0m", end="")  #! reset color
                return res

            return inner

        return outer

    def compute_candidates(self):
        for cell in self.mat.flatten():
            if cell.val:  # ? not empty
                continue
            no = self.region(cell, by=Cell.eq_by_val)
            Cell.__eq__ = Cell.eq_by_val
            cell._cands = self.NUMS - no

    @info_deco
    def naked_single(self):
        return self.naked_multiple(1)

    @info_deco
    def naked_pair(self):
        return self.naked_multiple(2)

    @info_deco
    def naked_triple(self):
        return self.naked_multiple(3)

    @info_deco
    def naked_quad(self):
        return self.naked_multiple(4)

    def naked_multiple(self, ple: int):
        low = (ple >= 2) + 1
        popped = []
        for n, func in itertools.product(range(9), (self.subgrid, self.row, self.col)):
            q = func(n)
            for cells in itertools.combinations(
                filter(lambda cell: low <= len(cell) <= ple, q), ple
            ):
                if (cands := set().union(*cells)) and len(cands) == ple:
                    if ple == 1:
                        for cell in cells:
                            self.update_region_for_cell(cell, *cands)
                            popped.append(cell)
                        continue
                    for cell in q:
                        Cell.__eq__ = Cell.eq_by_loc
                        if cell not in cells and (cell.cands & cands):
                            cell.remove_cands(*cands)
                            popped.append(cell)
        return popped

    @info_deco
    def hidden_single(self):
        return self.hidden_multiple(1)

    @info_deco
    def hidden_pair(self):
        return self.hidden_multiple(2)

    @info_deco
    def hidden_triple(self):
        return self.hidden_multiple(3)

    @info_deco
    def hidden_quad(self):
        return self.hidden_multiple(4)

    def hidden_multiple(self, ple):
        popped = []
        Cell.__eq__ = Cell.eq_by_loc
        for n, func in itertools.product(range(9), (self.subgrid, self.row, self.col)):
            dic = defaultdict(list)
            for cell in func(n):
                for cand in cell.cands:
                    dic[cand].append(cell)
            for items in itertools.combinations(dic.items(), ple):
                cands, list_of_cells = zip(*items)
                list_of_cells: list[list[Cell]]
                cands = set(cands)
                Cell.__eq__ = Cell.eq_by_loc
                cells = set().union(*list_of_cells)
                list_of_cands = [cell.cands for cell in cells]
                all_cands = set.union(*list_of_cands)
                if len(cells) == ple and cands <= all_cands:
                    for cell in cells:
                        if ple == 1:
                            self.update_region_for_cell(cell, *cands)
                        elif cell.cands != (new_cands := cands & cell.cands):
                            cell._cands = new_cands
                        else:
                            continue
                        popped.append(cell)
                if ple > 1 and popped:
                    return popped
        return popped

    @info_deco
    def pointing_intersection(self):
        popped = []
        for n in range(9):
            dic = defaultdict(set)
            for cell in self.subgrid(n):
                for cand in cell.cands:
                    dic[cand].add(cell.loc)
            for cand, locs in dic.items():
                rows, cols = map(set, zip(*locs))
                if len(rows) == 1:
                    q = self.row(rows.pop())
                elif len(cols) == 1:
                    q = self.col(cols.pop())
                else:
                    continue
                for cell in q:
                    Cell.__eq__ = Cell.eq_by_loc
                    if cell in self.subgrid(n):
                        continue
                    if cand in cell.cands:
                        cell.remove_cands(cand)
                        popped.append(cell)
        return popped

    @info_deco
    def claiming_intersection(self):
        popped = []
        for n, (i, func) in itertools.product(
            range(9), enumerate((self.row, self.col))
        ):
            dic = defaultdict(set)
            for cell in func(n):
                for cand in cell.cands:
                    dic[cand].add(cell.z)
            for cand, inds in dic.items():
                if len(inds) != 1:
                    continue
                for cell in self.subgrid(inds.pop()):
                    if cand in cell.cands and cell.loc[i] != n:
                        cell.remove_cands(cand)
                        popped.append(cell)
        return popped

    @info_deco
    def X_wing(self):
        return self.fish_multiple(2, finned=False)

    @info_deco
    def Swordfish(self):
        return self.fish_multiple(3, finned=False)

    @info_deco
    def Jellyfish(self):
        return self.fish_multiple(4, finned=False)

    @info_deco
    def finned_X_wing(self):
        return self.fish_multiple(2, finned=True)

    @info_deco
    def finned_Swordfish(self):
        return self.fish_multiple(3, finned=True)

    @info_deco
    def finned_Jellyfish(self):
        return self.fish_multiple(4, finned=True)

    @info_deco
    def XY_wing(self):
        popped = []
        for pivot in self.mat.flatten():
            pivot: Cell
            if len(pivot.cands) != 2:  # must be bi-valued
                continue
            for cell1, cell2 in itertools.combinations(
                filter(
                    lambda cell: len(cell.cands) == 2,  # must be bi-valued
                    self.region(pivot, by=Cell.eq_by_loc),
                ),
                2,
            ):
                cands1, cands2 = cell1.cands.copy(), cell2.cands.copy()
                if len(ZZ := cands1 & cands2) != 1:
                    continue
                if (Z := ZZ.pop()) in pivot.cands:
                    continue
                if cands1 | cands2 == {Z} | pivot.cands:
                    reg1 = self.region(cell1, by=Cell.eq_by_loc)
                    reg2 = self.region(cell2, by=Cell.eq_by_loc)
                    Cell.__eq__ = Cell.eq_by_loc
                    for buddy in (reg1 & reg2) - {cell1, cell2, pivot}:
                        if Z in buddy.cands:
                            buddy.remove_cands(Z)
                            popped.append(buddy)
                if popped:
                    return popped

    @info_deco
    def XYZ_wing(self):
        popped = []
        tris: set[Cell] = set()
        bis: set[Cell] = set()
        Cell.__eq__ = Cell.eq_by_loc
        for cell in self.mat.flatten():
            match len(cell.cands):
                case 2:
                    bis.add(cell)
                case 3:
                    tris.add(cell)
        for tri, wings in itertools.product(tris, itertools.combinations(bis, 2)):
            trio_cells: set[Cell] = {tri, *wings}
            ccands = tuple(c.cands for c in trio_cells)
            if len(ZZ := set.intersection(*ccands)) != 1:
                continue
            if len(set.union(*ccands)) != 3:
                continue
            Z = ZZ.pop()
            regdic: dict[Cell, set[Cell]] = {
                c: self.region(c, by=Cell.eq_by_loc) for c in trio_cells
            }
            if sum(tri in regdic[w] for w in wings) != 2:
                continue
            buddies = set.intersection(*regdic.values()) - trio_cells
            for buddy in buddies:
                if Z in buddy.cands:
                    buddy.remove_cands(Z)
                    popped.append(buddy)
            if popped:
                return popped

    def update_region_for_cell(self, cell: Cell, val: int):
        cell.val = val
        for reg_cell in self.region(cell, by=Cell.eq_by_loc):
            reg_cell.remove_cands(val)

    @property
    def solved(self):
        return all(c.val for c in self.mat.flat)

    @property
    def good(self):
        return all(c.val or c._cands for c in self.mat.flat)

    def solve(self, show=True):
        while not self.solved and self.good:
            didnt_break = True
            for alg in self.algorithms:
                if not show and ("fish" in alg.__name__ or "X_" in alg.__name__):
                    continue
                if alg(show)(self):
                    didnt_break = False
                    break
            if didnt_break:
                break
        return self

    @info_deco
    def backtracking(self):
        for cell in sorted(
            self.mat.flat,
            key=lambda c: (len(c.cands) or float("inf"), *c.loc),  #! mrv
        ):
            cell: Cell
            for try_cand in sorted(cell._cands):
                sdk_copy = self.copy()
                self.LAST.append((cell.loc, try_cand))
                cell_copy: Cell = sdk_copy.mat[cell.loc]
                sdk_copy.update_region_for_cell(cell_copy, try_cand)
                if sdk_copy.good and sdk_copy.solve(show=False).solved:
                    popped = []
                    for loc, try_cand in self.LAST:
                        cell = self.mat[loc]
                        self.update_region_for_cell(cell, try_cand)
                        popped.append(cell)
                    return popped
                self.LAST.remove((cell.loc, try_cand))

    algorithms = [
        hidden_single,  ##* 1
        naked_single,  ##* 2
        pointing_intersection,  ##* 3
        naked_pair,  ##* 4
        claiming_intersection,  ##* 5
        naked_triple,  ##* 6
        hidden_pair,  ##* 7
        hidden_triple,  ##* 15
        naked_quad,  ##* 14
        XY_wing,  ##* 12
        XYZ_wing,  ##* 13
        hidden_quad,  ##* 17
        # X_wing,  ##? 8
        # finned_X_wing,  ##? 10
        # Swordfish,  ##? 11
        # finned_Swordfish,  ##? 9
        # Jellyfish,  ##? 16
        # finned_Jellyfish,  ##? 18
        backtracking,  ##! last option if not solved logically
    ]
    FASTFISH = True
    LAST = []


def main():
    import time

    seeds = "easy medium hard extreme unfair".title().split()
    basedir = Path(__file__).parent
    Sudoku.fish_multiple = (
        Sudoku.fast_fish_multiple if Sudoku.FASTFISH else Sudoku.slow_fish_multiple
    )
    while True:
        print(
            "1. input from file",
            "2. input from seeds",
            "3. input from keyboard",
            "4. test-seeds",
            "else. exit",
            sep="\n",
        )
        match input():
            case "1":  # * open file
                fname = filedialog.askopenfilename(
                    initialdir=basedir,
                    initialfile=basedir / "test-seeds/very-hard/sashimi.txt.",
                )
                if not fname:
                    continue
                fpath = Path(fname or "")
                if not fpath.is_file():
                    continue
                try:
                    sdk, text = Sudoku.from_file(fpath)
                except:
                    traceback.print_exc()
                    continue
            case "2":  # * from seed
                print(
                    *(f"{ii}. {ss}" for ii, ss in enumerate(seeds, 1)),
                    "r. Random",
                    "else. exit",
                    sep="\n",
                )
                try:
                    if (inp := input()) == "r":
                        inp = random.choice("12345" * 3)
                    si = int(inp) - 1
                    if not (len(seeds) > si >= 0):
                        raise ValueError()
                except ValueError:
                    continue
                seed = seeds[si]
                print("selected seed", seed)
                with (basedir / "seeds" / seed).with_suffix(".seed").open() as fp:
                    lines = fp.seek(0, 2) // 82
                    pos = (line := random.randint(0, lines - 1)) * 82
                    fp.seek(pos)
                    print("selected line", line)
                    sdk, text = Sudoku.from_string(fp.readline().strip())
            case "3":  # ? from kb
                inp = input("input 81 string (or exit)")
                if inp == "exit":
                    continue
                try:
                    sdk, text = Sudoku.from_string(inp)
                except:
                    traceback.print_exc()
                    continue
            case "4":  # ? test-seeds
                tt = time.time()
                for file in basedir.glob("test-seeds/**/*.txt*"):
                    sdk, _ = Sudoku.from_file(file)
                    print(file.relative_to(basedir))
                    print("result:", "-" * 45, sdk.solve(show=1), "", sep="\n")
                print(time.time() - tt)
                continue
            case _:
                return
        # solving
        print(text)
        tt = time.time()
        print("result:", "-" * 45, sdk.solve(show=1), "", sep="\n")
        print(time.time() - tt)
        for c in sdk.mat.flat:
            c.print_wow_with_check()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
