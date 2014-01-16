# Parser modes
STRICT = 'STRICT'
FUZZY = 'FUZZY'
mode = STRICT

# Parser states / line types
ACTION = 'ACTION'
CONTINUED = 'CONTINUED'
DIALOG = 'DIALOG'
DIALOG_HEADER = 'DIALOG_HEADER'
DIRECTION = 'DIRECTION'
EMPTY = 'EMPTY'
ERROR = 'ERROR'
FRONT = 'FRONT'
PAGE_NUM = 'PAGE_NUM'
RESUMED = 'RESUMED'
SCENE_HEADING = 'SCENE_HEADING'

# Noun types
CHARACTER = 'CHARACTER'
THING = 'THING'
LOCATION = 'LOCATION'

# Interaction types
SETTING = 'SETTING' # When a noun is mentioned in a scene whose
                    # setting is defined by the location for the
                    # scene. The location will be the first party in
                    # the interaction.
DISCUSS = 'DISCUSS' # When two characters have lines in the same
                    # dialog block.  Order is not defined.
MENTION = 'MENTION' # When the first party speaks the name of the
                    # second in dialog.
APPEAR  = 'APPEAR'  # When two nouns appear in the same sentence.
