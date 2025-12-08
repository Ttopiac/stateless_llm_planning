(define (domain precondition-with-type-mismatch)
        (:action move
         :parameters (?x ?y)
         :precondition (= ?x ?y)))