(define (domain invalid-problems-domain)
        (:requirements :typing)
        (:types room person)
        (:constants lobby - room)
        (:predicates (at ?p - person ?r - room))
        (:action move
         :parameters (?p - person)))