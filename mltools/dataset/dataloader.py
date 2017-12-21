import logging

logger=logging.getLogger(__name__)

from pathlib import Path
from itertools import groupby,islice

def path_visitor(path,names,found):
    '''
        Recursively traverses directory and looks for given files.
    '''
    for name in names:
        if name==path.parts[-1]:
            logger.debug(f"Found file {str(path)}")
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
    logger.info(f"Found {len(found)} data files under {str(entry_dir)}.")
    return found

def group_filenames(groupnames):
    '''
        Splits given list of file names into a dict of given groups. Should
        be used in case data is put in files named according to labels.
    '''
    assert type(groupnames)==list, "groupnames should be a list."
#    assert type(filenames)==list, "filenames should be a list."

    def namer(filename):
        for name in groupnames:
            if name.lower() in filename.lower():
                return name
        return "UNKNOWN"

    def grouper(filenames):
        #groups=groupby(map(namer,filenames),key=lambda x:x[0])
        groups=groupby(filenames,key=namer)
        _groups={}
        for key,iterator in groups:
            _groups[key]=list(iterator)
        return _groups

    return grouper

def split_files_into_datasets(groups):
    '''
        Splits the given groups of labelled iterators into training, test and
        validation sets.
    '''
    assert type(groups)==dict,"Groups should be dict of format {label1:iterator1,label2:iterator2}"

    for label,iterator in groups.items():
        iter_len=len(list(iterator))
        train_len=0.5*iter_len
        test_len=val_len=0.25*iter_len
        groups[label]={"train":islice(iterator,0,train_len),
                        "val":islice(iterator,train_len,train_len+val_len),
                        "test":islice(iterator,train_len+val_len,None)}

    return groups

