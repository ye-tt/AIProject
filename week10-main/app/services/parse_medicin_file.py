from __future__ import annotations

from pathlib import Path
import pandas as pd


def parse_directory(
    directory: Path, suffix: str = ".csv"
) -> list():
    """Parse all text files in a directory into document chunks."""
    processed_data = list()
    all_datas = list()
    for file_path in directory.glob(f"*{suffix}"):
        datas = pd.read_csv(file_path, encoding='gb18030')
        datas_list = datas.sample(frac=0.1, random_state=42).to_dict(orient='records')
        # datas_list = datas.to_dict(orient='records')
        all_datas.extend(datas_list)
    # print(all_datas[0:5])
    return all_datas

if __name__ == "__main__":
    print(Path(__file__).parent.parent.parent/ "data"/"me")
    all_datas = parse_directory(
         directory=Path(__file__).parent.parent.parent/ "data"/"me",
         suffix=".csv"
     )
