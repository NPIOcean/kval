'''
Functions prompting the user for input.

Intended for jupyter notebooks, won't work outside

(consider making a parallel library for ipython-type inputs?)
'''

import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

## Enter attributes using inteactive widgets
# Can probably refactor this..
    
def set_attr_textbox(D, attr, rows=1, cols=50, initial_value = None):
    """
    Create a widget for setting a text attribute with a text box.

    Parameters:
        D (dict): The dictionary to update with the entered text for the attribute.
        attr (str): The attribute name.
        rows (int): The number of rows for the text box (default is 1).
        cols (int): The number of columns for the text box (default is 50).

    Returns:
        dict: The updated dictionary with the entered text for the attribute.
    """

    def _on_button_click(b, D, attr, text_input):
        clear_output(wait=True)
        D.attrs[attr] = text_input.value

        # Close the widget
        vbox.close()

        
    prompt1 = (
        f"Enter the desired value for the "
        f"<span style='color: blue; font-weight: bold;'>{attr}</span>"
        ' global attribute:'
    )
    prompt2 = '(Click "Submit" button to apply to dataset)'
    text_input = widgets.Textarea(rows=rows, cols=cols, 
                                  value = initial_value)
    
    # Create a button widget
    enter_button = widgets.Button(description="Submit")
    
    # Attach the function to the button's click event using a lambda function
    enter_button.on_click(
        lambda b: _on_button_click(b, D, attr, text_input))
    
    # Create a VBox to arrange the prompt, text box, and button
    vbox = widgets.VBox([widgets.HTML(value=prompt1), widgets.HTML(value=prompt2),
                         text_input, enter_button])
    
    # Display the VBox
    display(vbox)
    
    return D  # Return the modified dictionary


def set_var_attr_textbox(D, varnm, attr, rows=1, cols=50, initial_value = None):
    """
    Create a widget for setting a variable text attribute with a text box.

    Parameters:
        D (dict): The dictionary containing the variable and its attributes.
        varnm (str): The variable name.
        attr (str): The attribute name.
        rows (int): The number of rows for the text box (default is 1).
        cols (int): The number of columns for the text box (default is 50).

    Returns:
        dict: The updated dictionary with the entered text for the attribute.
    """

    def on_button_click(b):
        clear_output(wait=True)
        D[varnm].attrs[attr] = text_input.value

        # Close the widget
        vbox.close()

    prompt1 = (
        f"Enter the desired value for the "
        f"<span style='color: blue; font-weight: bold;'>{attr}</span>"
        f' attribute of variable {varnm}:'
    )
    prompt2 = '(Click "Submit" button to apply to dataset)'

    text_input = widgets.Textarea(rows=rows, cols=cols, 
                                  value = initial_value)
    
    # Create a button widget
    enter_button = widgets.Button(description="Submit")
    
    # Attach the function to the button's click event using a lambda function
    enter_button.on_click(on_button_click)
    
    # Create a VBox to arrange the prompt, text box, and button
    vbox = widgets.VBox([widgets.HTML(value=prompt1), widgets.HTML(value=prompt2),
                         text_input, enter_button])
    
    # Display the VBox
    display(vbox)
    
    return D  # Return the modified dictionary



def set_attr_pulldown(D, attr, options_with_comments, initial_value = None):
    """
    Create a widget for setting an attribute with a pulldown menu and optional comments.

    Parameters:
        D (dict): The dictionary to update with the selected attribute value.
        attr (str): The attribute name.
        options_with_comments (list or list of lists): List of options for the pulldown menu.
            If comments are available, it should be a list of lists where each inner list contains
            the option and its associated comment. If no comments, it can be a simple list of options.

    Returns:
        dict: The updated dictionary with the selected attribute value.

    """
    def on_button_click(b):
        selected_value = dropdown.value
        if selected_value is not None:
            D.attrs[attr] = selected_value
        else:
            # Handle the case when no option is selected, e.g., show an error message
            pass

        # Close the widget
        vbox.close()

        
    if options_with_comments and all(isinstance(opt, list) for opt in options_with_comments):

        allowed_options, option_comments = zip(*options_with_comments)

        dropdown = widgets.Dropdown(options=allowed_options, value=initial_value, 
            style={'description_width': 'initial', 'width': '700px'})
        
        dropdown_label = widgets.HTML(f"Attribute: <b style='color: blue;'>{attr}</b>")
        comment_box = widgets.Textarea(value='', disabled=True, layout=widgets.Layout(width='700px'))
        comment_box_label = widgets.Label("Description:")

        def update_comment_box(change):
            selected_option = dropdown.value
            if option_comments and selected_option in allowed_options:
                comment_box.value = option_comments[allowed_options.index(selected_option)]
            else:
                comment_box.value = ""

        dropdown.observe(update_comment_box, names='value')

        vbox = widgets.VBox([dropdown_label, widgets.HBox([dropdown]), widgets.HBox([widgets.VBox([comment_box_label, comment_box])])])

        submit_button = widgets.Button(description="Submit")

        submit_button.on_click(on_button_click)

        vbox.children += (submit_button,)

    else:
        dropdown = widgets.Dropdown(options=options_with_comments, value=None, style={'description_width': 'initial', 'width': '300px'})
        dropdown_label = widgets.HTML(f"Attribute: <b style='color: blue;'>{attr}</b>")
        submit_button = widgets.Button(description="Submit")
        submit_button.on_click(on_button_click)

        vbox = widgets.VBox([dropdown_label, widgets.HBox([dropdown]), submit_button])

    display(vbox)

    return D




def set_var_attr_pulldown(D, varnm, attr, options_with_comments, initial_value = None):
    """
    Create a widget for setting a variable attribute with a pulldown menu and optional comments.

    Parameters:
        D (dict): The dictionary containing the variable and its attributes.
        varnm (str): The variable name.
        attr (str): The attribute name.
        options_with_comments (list or list of lists): List of options for the pulldown menu.
            If comments are available, it should be a list of lists where each inner list contains
            the option and its associated comment. If no comments, it can be a simple list of options.

    Returns:
        dict: The updated dictionary with the selected attribute value.

    """
    def on_button_click(b):
        selected_value = dropdown.value
        if selected_value is not None:
            D[varnm].attrs[attr] = selected_value
        else:
            # Handle the case when no option is selected, e.g., show an error message
            pass

        # Close the widget
        vbox.close()

        
    if options_with_comments and all(isinstance(opt, list) for opt in options_with_comments):
        allowed_options, option_comments = zip(*options_with_comments)

        dropdown = widgets.Dropdown(options=allowed_options, value=initial_value, 
            style={'description_width': 'initial', 'width': '700px'}, 

                                  initial_value = initial_value)
        dropdown_label = widgets.HTML(f"Attribute: <b style='color: blue;'>{attr}</b>")
        comment_box = widgets.Textarea(value='', disabled=True, layout=widgets.Layout(width='700px'))
        comment_box_label = widgets.Label("Description:")

        def update_comment_box(change):
            selected_option = dropdown.value
            if option_comments and selected_option in allowed_options:
                comment_box.value = option_comments[allowed_options.index(selected_option)]
            else:
                comment_box.value = ""

        dropdown.observe(update_comment_box, names='value')

        vbox = widgets.VBox([dropdown_label, widgets.HBox([dropdown]), widgets.HBox([widgets.VBox([comment_box_label, comment_box])])])

        submit_button = widgets.Button(description="Submit")

        submit_button.on_click(on_button_click)

        vbox.children += (submit_button,)

    else:
        dropdown = widgets.Dropdown(options=options_with_comments, value=None, style={'description_width': 'initial', 'width': '300px'})
        dropdown_label = widgets.HTML(f"Attribute: <b style='color: blue;'>{attr}</b>")
        submit_button = widgets.Button(description="Submit")
        submit_button.on_click(on_button_click)

        vbox = widgets.VBox([dropdown_label, widgets.HBox([dropdown]), submit_button])

    display(vbox)

    return D

