import openai 
import streamlit as st 
import datetime 
import pytz 

def get_current_time():
    return datetime.datetime.now(pytz.timezone('US/Pacific'))


class open_ai:

    class BadRequest(Exception):
        pass


    def __init__(self, api_key, restart_sequence, stop_sequence):
        self.api_key = api_key
        openai.api_key = api_key 
        self.stop_sequence = stop_sequence
        self.restart_sequence = restart_sequence


    def _invoke_call(self, call_string):
        """Generic function to invoke openai calls"""
        try:
            result = eval(f"{call_string}")
            return result      

        except Exception as e: 
            raise 

    def get_moderation(self, user_message):
        """Main function to get moderation on a user message"""
        get_moderation_call_string = ("""openai.Moderation.create(input="{0}")""".format(user_message))

        try:
            moderation = self._invoke_call(get_moderation_call_string)
            moderation_result = moderation['results'][0]
            flagged_categories = [category for category, value in moderation_result['categories'].items() if value]

            return {'flagged': moderation_result['flagged'], 'flagged_categories':flagged_categories}
        except Exception as e:
            raise 


    def get_models(self):
        """Main function to get models that the key has access to """
        try:
            return self._invoke_call("openai.Model.list()")
        except Exception as e:
            raise


    def get_ai_response(self, model_config_dict, init_prompt_msg, messages):

        submit_messages = [{'role':'system','message':init_prompt_msg,'current_date':get_current_time()}]+ messages

        new_messages = [] 
        bot_message = ''
        total_tokens = 0

        if model_config_dict['model'] in ('gpt-3.5-turbo', 'gpt-4'):
            try:
                response = self._get_chat_completion(model_config_dict, submit_messages)
                bot_message = response['choices'][0]['message']['content']
                total_tokens = response['usage']['total_tokens']
            except Exception as e:
                raise 
        else:
            try:
                response = self._get_completion(model_config_dict, submit_messages)
                bot_message = response['choices'][0]['text']
                total_tokens = response['usage']['total_tokens']
            except Exception as e:
                raise
        
        new_messages = messages + [{'role':'assistant','message':bot_message.strip(),'created_date':get_current_time()}]

        return {'messages':new_messages, 'total_tokens':total_tokens}   


    def _get_chat_completion(self, model_config_dict, messages):
        model_config_validated = self._validate_model_config(model_config_dict)
        oai_messages = self._messages_to_oai_messages(messages)

        if model_config_validated:
            get_completion_call_string = (
            """openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages={0},
                temperature={1},
                max_tokens={2},
                top_p={3},
                frequency_penalty={4},
                presence_penalty={5},
                stop=['{6}']
                )""").format(
                    #model_config_dict['model'],
                    oai_messages,
                    model_config_dict['temperature'],
                    model_config_dict['max_tokens'],
                    model_config_dict['top_p'],
                    model_config_dict['frequency_penalty'],
                    model_config_dict['presence_penalty'],
                    self.stop_sequence
                )            
            
            try:
                completions = self._invoke_call(get_completion_call_string)
                return completions
            except Exception as e:
                raise 
        else:
            if not model_config_validated:
                raise self.BadRequest("Bad Request. model_config_dict missing required fields")


    def _get_completion(self, model_config_dict, messages):
        model_config_validated = self._validate_model_config(model_config_dict)
        oai_message = self._messages_to_oai_prompt_str(messages)

        if model_config_validated:
            get_completion_call_string = (
            """openai.Completion.create(
                model="{0}",
                prompt="{1}",
                temperature={2},
                max_tokens={3},
                top_p={4},
                frequency_penalty={5},
                presence_penalty={6},
                stop=['{7}']
                )""").format(
                    model_config_dict['model'],
                    oai_message,
                    model_config_dict['temperature'],
                    model_config_dict['max_tokens'],
                    model_config_dict['top_p'],
                    model_config_dict['frequency_penalty'],
                    model_config_dict['presence_penalty'],
                    self.stop_sequence
                )            
            
            try:
                completions = self._invoke_call(get_completion_call_string)
                return completions
            except Exception as e:
                raise 
        else:
            if not model_config_validated:
                raise self.BadRequest("Bad Request. model_config_dict missing required fields")
            

    # helper functions 
    def _validate_model_config(self, model_config_dict):
        required_fields = ['model', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']

        for field in required_fields:
            if field not in model_config_dict:
                raise self.ClientRequestError("Bad model configuration request") 
        return True


    def _messages_to_oai_prompt_str(self, messages):
        msg_string = ""
        for message in messages:
            if message['role'] == 'user' or message['role'] == 'system':
                msg_string += message['message'].replace("\"","'") + self.stop_sequence
            else:
                msg_string += message['message'].replace("\"","'") + self.restart_sequence
        return msg_string


    def _messages_to_oai_messages(self, messages):
        oai_messages = []
        if len(messages) > 0:
            for message in messages:
                oai_messages.append({'role':message['role'], 'content':message['message']})
        return oai_messages 

