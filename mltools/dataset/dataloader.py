import logging

logger=logging.getLogger(__name__)

from pathlib import Path

def path_visitor(path,names,found):
    for name in names:
        if name==path.parts[-1]:
            found.append(str(path))
    if not path.is_dir():
        return
    for _path in path.iterdir():
        path_visitor(_path,names,found)

def get_data_files(entry_dir,filenames):
    '''
        Recursively traverses entry_dir to find all filenames.
        Returns full path of found filenames.
    '''
    assert type(entry_dir)==str, "Entry directory should be a string."
    entry_dir=Path(entry_dir)
    found=[]
    path_visitor(entry_dir,filenames,found)
    return found
