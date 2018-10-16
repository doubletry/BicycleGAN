set -ex
# models
RESULTS_DIR='./results/colourblindness'
#G_PATH='./pretrained_models/edges2shoes_net_G.pth'
#E_PATH='./pretrained_models/edges2shoes_net_E.pth'

# dataset
CLASS='colourblindness'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
HOW_MANY=771 # number of input images duirng test
NUM_SAMPLES=1 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/colourblindness  \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./checkpoints/ \
  --name ${CLASS} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip
