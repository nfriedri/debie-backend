import logging


def return_debiasing(methods, target1, target2, arg1, arg2):
    logging.info("APP-DE: Forwarding to related definitions")
    if methods is None:
        return return_gbdd(target1, target2, arg1, arg2)
    if methods == 'gbdd':
        return return_gbdd(target1, target2, arg1, arg2)
    if methods == 'debiasNet':
        return return_debias_net(target1, target2, arg1, arg2)
    if methods == 'bam':
        return return_bam(target1, target2, arg1, arg2)
    if methods == 'gbddxbam':
        return return_gbdd_bam(target1, target2, arg1, arg2)
    if methods == 'bamxgbdd':
        return return_bam_gbdd(target1, target2, arg1, arg2)
    if methods == 'gbddxdebiasNet':
        return return_gbdd_debias_net(target1, target2, arg1, arg2)
    return 400


def return_gbdd(target1, target2, arg1, arg2):
    logging.info("APP-DE: Starting GBDD debiasing")
    return 0


def return_bam(target1, target2, arg1, arg2):
    logging.info("APP-DE: Starting BAM debiasing")
    return 0


def return_debias_net(target1, target2, arg1, arg2):
    logging.info("APP-DE: Starting DebiasNet debiasing")
    return 0


def return_gbdd_bam(target1, target2, arg1, arg2):
    logging.info("APP-DE: Starting GBDD°BAM debiasing")
    return 0


def return_bam_gbdd(target1, target2, arg1, arg2):
    logging.info("APP-DE: Starting BAM°GBDD debiasing")
    return 0


def return_gbdd_debias_net(target1, target2, arg1, arg2):
    logging.info("APP-DE: Starting GBDD°DebiasNet debiasing")
    return 0
