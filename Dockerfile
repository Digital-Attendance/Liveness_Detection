FROM public.ecr.aws/lambda/python:3.12

RUN microdnf update -y && \
    microdnf install -y mesa-libGL

# Copy your application code to the Lambda task root directory
COPY . ${LAMBDA_TASK_ROOT}

# Upgrade pip and install dependencies with an increased timeout
RUN pip install --upgrade pip && \
    pip install --default-timeout=60 -r requirements.txt

# Set the CMD to your Lambda function handler.
# Adjust "app.lambda_handler" if your module or handler function name differs.
CMD ["app.lambda_handler"]
