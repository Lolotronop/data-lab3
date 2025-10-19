uvx dvc stage add -n resnet \
                        -p resnet.seed,resnet.epochs \
                        -d resnet.py -d ignored/data \
                        -o ignored/models/resnet.h5 \
                        uv run resnet.py
