(define (domain predicate-with-type-mismatch-via-constant)
        (:requirements :typing)
        (:types cool beans)
        (:constants my-bean - beans)
        (:predicates (wow ?x - cool))
        (:action move
         :parameters ()
         :precondition (wow my-bean)))