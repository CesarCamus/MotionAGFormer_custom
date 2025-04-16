MODEL_PATH="$1"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <path-to-model.onnx>"
    exit 1
fi

DIRNAME=$(basename "$(dirname "$MODEL_PATH")")
FILENAME=$(basename "$MODEL_PATH")
MODEL_DIR=$(dirname "$MODEL_PATH")
BASENAME="${FILENAME%.*}"
LOCAL_PLAN="${MODEL_DIR}/${BASENAME}.plan"

REMOTE_USER="andrew"
REMOTE_HOST="192.168.1.227"
REMOTE_PASS="andrew"
REMOTE_DIR="/home/andrew/Documents/models/POSE_3D/${DIRNAME}"

echo "Converting: ${MODEL_PATH}"

# Create remote directory and copy model
sshpass -p "${REMOTE_PASS}" ssh -o StrictHostKeyChecking=no \
    ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}"

sshpass -p "${REMOTE_PASS}" scp -o StrictHostKeyChecking=no "${MODEL_PATH}" \
    ${REMOTE_USER}@${REMOTE_HOST}:"${REMOTE_DIR}/motionagformer.onnx"

# Convert to TensorRT engine
sshpass -p "${REMOTE_PASS}" ssh -o StrictHostKeyChecking=no \
    ${REMOTE_USER}@${REMOTE_HOST} << EOF
    set -e  # Exit if any command fails
    echo "Converting ONNX model to TensorRT..."
    /usr/src/tensorrt/bin/trtexec --onnx=${REMOTE_DIR}/motionagformer.onnx \
    --minShapes=input:1x243x17x3 \
    --optShapes=input:4x243x17x3 \
    --maxShapes=input:8x243x17x3 \
    --fp16 \
    --workspace=8182 --saveEngine=${REMOTE_DIR}/model.engine
    echo "Conversion completed: ${REMOTE_DIR}/model.engine"
EOF

# Copy engine back to local
sshpass -p "${REMOTE_PASS}" scp -o StrictHostKeyChecking=no \
    ${REMOTE_USER}@${REMOTE_HOST}:"${REMOTE_DIR}/model.engine" \
    "${LOCAL_PLAN}"

echo "TensorRT engine saved to: ${LOCAL_PLAN}"

# -minShapes=images:1x3x384x288 \
#     --optShapes=images:4x3x384x288 \
#     --maxShapes=images:8x3x384x288
