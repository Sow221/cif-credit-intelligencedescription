"""Package des pipelines MLOps.

La réévaluation périodique du champion se fait via ``pipelines.definitions.lending_club_job``
(planifié, voir ``lending_club_schedule``) — pas via un module séparé : un seul mécanisme de
promotion (``models.promotion.promote_if_better``), jamais deux implémentations divergentes.
"""
