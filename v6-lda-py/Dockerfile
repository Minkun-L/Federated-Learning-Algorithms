FROM harbor2.vantage6.ai/infrastructure/algorithm-base

ARG PKG_NAME="v6-lda-py"

COPY . /app
RUN pip install /app

ENV PKG_NAME=${PKG_NAME}

CMD python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"
