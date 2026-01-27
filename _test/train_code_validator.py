import glob
import os

# check there are exactly 4 train scripts in code/ and they have the correct names
train_scripts = glob.glob("code/*.sh")
train_script_names = [os.path.basename(path) for path in train_scripts]

if len(train_scripts) < 4:
    print(
        f"Too few training scripts. You should have one per model. (You have {len(train_scripts)} in total.)"
    )
    exit()
elif len(train_scripts) > 4:
    print(
        f"Too many training scripts. You should have one per model. (You have {len(train_scripts)} in total.)"
    )
    exit()
for model in ["dpo", "mcqa", "quantized", "rag"]:
    if f"train_{model}.sh" not in train_script_names:
        print(f"Missing train script for {model}. Or it has a wrong name.")
        exit()

# check there are exactly 4 directories in code/ and they have the correct names
code_subdirs = [os.path.normpath(path) for path in glob.glob("code/*/")]
code_subdir_names = [os.path.basename(path) for path in code_subdirs]

if len(code_subdirs) > 4:
    print(
        f"Too many directories in code/. You should have one per model. (You have {len(code_subdirs)} in total.)"
    )
    exit()
elif len(code_subdirs) < 4:
    print(
        f"Too few directories in code/. You should have one per model. (You have {len(code_subdirs)} in total.)"
    )
    exit()

# check there is one subdirectory for each train script
for model in ["dpo", "mcqa", "quantized", "rag"]:
    if f"train_{model}" not in code_subdir_names:
        print(f"Missing directory for train script {model}.")
        exit()

# check there is more than just .placeholder in each subdirectory
for subdir in code_subdirs:
    contents = glob.glob(os.path.join(subdir, "*"))
    if len(contents) == 0:
        print(f"Directory {subdir} is empty.")
        exit()
    elif len(contents) == 1 and contents[0].endswith(".placeholder"):
        print(f"Directory {subdir} contains only .placeholder.")
        exit()

print("The training code exists.")
